# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real McCulloch-Pitts backend loading and build contracts

"""Build and resolve every maintained exact McCulloch--Pitts execution lane."""

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

from sc_neurocore.accel import mcculloch_pitts as backends
from sc_neurocore.accel.mojo.isa_baseline import pin_isa

_REPOSITORY = Path(__file__).resolve().parents[1]
_COUNTS = np.asarray([0, 1, 2, 2, (1 << 31) - 1], dtype=np.int64)
_FLAGS = np.asarray([0, 0, 0, 1, 0], dtype=np.uint8)


def test_rust_batch_binding_stays_out_of_crate_root() -> None:
    """Keep Model33 implementation in its claimed per-neuron binding module."""
    crate_root = (_REPOSITORY / "engine/src/lib.rs").read_text(encoding="utf-8")
    registry = (_REPOSITORY / "engine/src/pyo3_neurons.rs").read_text(encoding="utf-8")
    binding = (_REPOSITORY / "engine/src/bindings/mcculloch_pitts.rs").read_text(encoding="utf-8")

    assert "mcculloch_pitts" not in crate_root
    assert '#[path = "bindings/mcculloch_pitts.rs"]' in registry
    assert "mcculloch_pitts_binding::register(m)?;" in registry
    assert "fn py_mcculloch_pitts_evaluate_batch" in binding


def test_rust_engine_package_reexports_exact_batch() -> None:
    """Reach the full signed-count batch through the installed package."""
    import sc_neurocore_engine

    assert backends._HAS_RUST
    assert callable(sc_neurocore_engine.py_mcculloch_pitts_evaluate_batch)
    events, event_count = sc_neurocore_engine.py_mcculloch_pitts_evaluate_batch(
        2,
        _COUNTS,
        _FLAGS,
    )
    assert np.asarray(events).tolist() == [0, 0, 1, 0, 1]
    assert event_count == 2


def test_rust_same_name_class_exposes_source_truth_tables() -> None:
    """The PyO3 class implements OR, AND, absolute veto and strict counts."""
    import sc_neurocore_engine

    logical_or = sc_neurocore_engine.McCullochPittsNeuron()
    assert [logical_or.step(count) for count in (0, 1, 2)] == [0, 1, 1]
    assert logical_or.step((1 << 31) - 1, True) == 0
    assert logical_or.get_state() == {}
    logical_or.reset()

    logical_and = sc_neurocore_engine.McCullochPittsNeuron(theta=2)
    assert [logical_and.step(count) for count in (0, 1, 2)] == [0, 0, 1]
    with pytest.raises(ValueError, match="theta must be an integer"):
        sc_neurocore_engine.McCullochPittsNeuron(theta=1.5)
    with pytest.raises(ValueError, match="excitatory_count must be an integer"):
        logical_or.step(-1)


def test_julia_loader_executes_committed_module() -> None:
    """Load the Julia source and execute the exact varying-input rule."""
    assert backends.ensure_julia_loaded()
    events, event_count = backends.evaluate_julia(2, _COUNTS, _FLAGS)
    assert events.tolist() == [0, 0, 1, 0, 1]
    assert event_count == 2


def test_go_c_shared_build_reproduces_committed_header(tmp_path: Path) -> None:
    """Build the cgo package and pin its generated public ABI byte-for-byte."""
    source = _REPOSITORY / "src/sc_neurocore/accel/go/neurons/mcculloch_pitts"
    output = tmp_path / "libmcculloch_pitts.so"
    subprocess.run(
        ["go", "build", "-buildmode=c-shared", "-o", str(output), "."],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    library = ctypes.CDLL(str(output))
    assert library.mcculloch_pitts_evaluate_c is not None
    assert output.with_suffix(".h").read_bytes() == (source / "libmcculloch_pitts.h").read_bytes()


def test_mojo_shared_build_exports_exact_batch(tmp_path: Path, mojo_cli: str) -> None:
    """Compile the maintained Mojo source and resolve its C ABI."""
    source = _REPOSITORY / "src/sc_neurocore/accel/mojo/kernels/mcculloch_pitts.mojo"
    output = tmp_path / "libmcculloch_pitts.so"
    subprocess.run(
        pin_isa([mojo_cli, "build", "--emit", "shared-lib", "-o", str(output), str(source)]),
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert ctypes.CDLL(str(output)).mcculloch_pitts_evaluate_c is not None


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
    assert reloaded._engine_evaluate is None

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
