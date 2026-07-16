# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real IQIF backend loading and build contracts

"""Build and resolve every maintained exact IQIF execution surface."""

from __future__ import annotations

import ctypes
import importlib
import importlib.util as importlib_util
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from sc_neurocore.accel import iqif as backends
from sc_neurocore.accel.mojo.isa_baseline import pin_isa

_REPOSITORY = Path(__file__).resolve().parents[1]
_CONTRACT = (128, 128, 200, 128, 1, 1, 255, 0, 15, 10)


def test_rust_engine_package_reexports_exact_batch() -> None:
    """Reach the full signed-integer batch through the installed package."""
    import sc_neurocore_engine

    assert backends._HAS_RUST
    assert callable(sc_neurocore_engine.py_iqif_simulate)
    trace, spikes, final_v = sc_neurocore_engine.py_iqif_simulate(*_CONTRACT)
    assert np.asarray(trace).tolist() == [
        138,
        146,
        153,
        159,
        165,
        170,
        176,
        183,
        190,
        198,
        207,
        217,
        229,
        242,
        128,
    ]
    assert spikes == 1
    assert final_v == 128


def test_rust_stateful_class_has_full_constructor_and_reset() -> None:
    """The same-name PyO3 class exposes and preserves every parameter."""
    import sc_neurocore_engine

    neuron = sc_neurocore_engine.IntegerQIFNeuron(
        v=100,
        v_rest=96,
        v_threshold=180,
        v_reset=120,
        a=3,
        b=5,
        v_max=240,
        v_min=4,
    )
    neuron.step(17)
    neuron.reset()
    state = neuron.get_state()
    assert state == {
        "v": 96,
        "v_rest": 96,
        "v_threshold": 180,
        "v_reset": 120,
        "a": 3,
        "b": 5,
        "v_max": 240,
        "v_min": 4,
    }
    with pytest.raises(ValueError, match="invalid IQIF"):
        sc_neurocore_engine.IntegerQIFNeuron(a=-1)


def test_julia_loader_executes_committed_module() -> None:
    """Load Julia source and execute the exact source prefix."""
    assert backends.ensure_julia_loaded()
    trace, spikes, final_v = backends.simulate_julia(*_CONTRACT)
    assert trace[-1] == final_v == 128
    assert spikes == 1


def test_go_c_shared_build_reproduces_committed_header(tmp_path: Path) -> None:
    """Build the cgo package and pin its generated public ABI."""
    source = _REPOSITORY / "src/sc_neurocore/accel/go/neurons/iqif"
    output = tmp_path / "libiqif.so"
    subprocess.run(
        ["go", "build", "-buildmode=c-shared", "-o", str(output), "."],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    library = ctypes.CDLL(str(output))
    assert library.iqif_simulate_c is not None
    assert output.with_suffix(".h").read_bytes() == (source / "libiqif.h").read_bytes()


def test_mojo_shared_build_exports_exact_batch(tmp_path: Path) -> None:
    """Compile the maintained Mojo source and resolve its C ABI."""
    source = _REPOSITORY / "src/sc_neurocore/accel/mojo/kernels/iqif.mojo"
    output = tmp_path / "libiqif.so"
    subprocess.run(
        pin_isa(["mojo", "build", "--emit", "shared-lib", "-o", str(output), str(source)]),
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert ctypes.CDLL(str(output)).iqif_simulate_c is not None


def test_staged_go_and_mojo_artifacts_reload_through_dispatcher() -> None:
    """Resolve both actual staged libraries through production loaders."""
    assert backends.ensure_go_loaded()
    assert backends.ensure_mojo_loaded()
    assert backends._go_lib is not None
    assert backends._mojo_lib is not None


def test_engine_import_failure_disables_only_optional_rust_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Module initialisation keeps the Python floor without the extension."""
    original_import_module = importlib.import_module

    def missing_engine(name: str) -> ModuleType:
        if name == "sc_neurocore_engine":
            raise ImportError("engine absent")
        return original_import_module(name)

    monkeypatch.setattr(importlib, "import_module", missing_engine)
    reloaded = importlib.reload(backends)
    assert reloaded._HAS_RUST is False
    assert reloaded._engine_simulate is None

    monkeypatch.setattr(importlib, "import_module", original_import_module)
    restored = importlib.reload(reloaded)
    assert restored._HAS_RUST is True


def test_julia_loader_fails_closed_for_absent_or_broken_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every Julia discovery failure remains a non-fatal unavailable result."""
    monkeypatch.setattr(backends, "_julia_module", None)
    monkeypatch.setattr(importlib_util, "find_spec", lambda _name: None)
    assert backends.ensure_julia_loaded() is False

    monkeypatch.setattr(importlib_util, "find_spec", lambda _name: object())
    monkeypatch.setattr(os.path, "isfile", lambda _path: False)
    assert backends.ensure_julia_loaded() is False

    monkeypatch.setattr(os.path, "isfile", lambda _path: True)

    def fail_import(_name: str) -> ModuleType:
        raise RuntimeError("Julia startup failed")

    monkeypatch.setattr(importlib, "import_module", fail_import)
    assert backends.ensure_julia_loaded() is False

    sentinel = object()
    monkeypatch.setattr(backends, "_julia_module", sentinel)
    assert backends.ensure_julia_loaded() is True


@pytest.mark.parametrize(
    ("library_attribute", "flag_attribute", "loader"),
    (
        ("_go_lib", "_HAS_GO", backends.ensure_go_loaded),
        ("_mojo_lib", "_HAS_MOJO", backends.ensure_mojo_loaded),
    ),
)
def test_c_loaders_fail_closed_without_valid_abi(
    library_attribute: str,
    flag_attribute: str,
    loader: Callable[[], bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing files, loader errors and absent symbols cannot claim readiness."""
    monkeypatch.setattr(backends, library_attribute, None)
    monkeypatch.setattr(backends, flag_attribute, False)
    monkeypatch.setattr(os.path, "isfile", lambda _path: False)
    assert loader() is False

    monkeypatch.setattr(os.path, "isfile", lambda _path: True)

    def fail_load(_path: str) -> object:
        raise OSError("invalid library")

    monkeypatch.setattr(ctypes, "CDLL", fail_load)
    assert loader() is False
    monkeypatch.setattr(ctypes, "CDLL", lambda _path: object())
    assert loader() is False

    sentinel = object()
    monkeypatch.setattr(backends, library_attribute, sentinel)
    assert loader() is True
