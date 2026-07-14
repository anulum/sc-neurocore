# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — real Poisson backend loading contracts

"""Build and resolve every maintained seeded Poisson execution surface."""

from __future__ import annotations

from collections.abc import Callable
import ctypes
import importlib
import importlib.util as importlib_util
import os
from pathlib import Path
import subprocess
from types import ModuleType

import numpy as np
import pytest

from sc_neurocore.accel import poisson as backends

_REPOSITORY = Path(__file__).resolve().parents[1]


def _contract() -> tuple[float, float, int, int, float]:
    return (250.0, 1.0, 0xACE1, 1, -1.0)


def test_rust_engine_package_reexports_the_seeded_batch() -> None:
    """Reach the production PyO3 batch through the installed package boundary."""
    import sc_neurocore_engine

    assert backends._HAS_RUST
    assert callable(sc_neurocore_engine.py_poisson_simulate)
    events, final_rng = sc_neurocore_engine.py_poisson_simulate(*_contract())
    assert np.asarray(events).tolist() == [1]
    assert 1 <= final_rng <= 0xFFFF


def test_rust_stateful_reset_replays_the_rng_state() -> None:
    """The same-name PyO3 class exposes and restores its explicit seed."""
    import sc_neurocore_engine

    neuron = sc_neurocore_engine.PoissonNeuron(rate_hz=250.0, dt_ms=1.0, seed=42)
    first = [neuron.step() for _ in range(100)]
    neuron.reset()
    state = neuron.get_state()
    assert state["rate_hz"] == 250.0
    assert state["dt_ms"] == 1.0
    assert state["rng_state"] == state["initial_seed"] == 42
    assert [neuron.step() for _ in range(100)] == first


def test_julia_loader_executes_the_committed_module() -> None:
    """Load Julia source and execute the complete rate/RNG batch."""
    assert backends.ensure_julia_loaded()
    events, final_rng = backends.simulate_julia(*_contract())
    assert events.tolist() == [1]
    assert 1 <= final_rng <= 0xFFFF


def test_go_c_shared_build_reproduces_the_committed_header(tmp_path: Path) -> None:
    """Build the cgo package and keep its public C ABI reproducible."""
    source = _REPOSITORY / "src/sc_neurocore/accel/go/neurons/poisson"
    output = tmp_path / "libpoisson.so"
    subprocess.run(
        ["go", "build", "-buildmode=c-shared", "-o", str(output), "."],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    library = ctypes.CDLL(str(output))
    assert library.poisson_simulate_c is not None
    assert output.with_suffix(".h").read_bytes() == (source / "libpoisson.h").read_bytes()


def test_mojo_shared_build_exports_the_real_batch_symbol(tmp_path: Path) -> None:
    """Compile the maintained Mojo source and resolve its exported C ABI."""
    source = _REPOSITORY / "src/sc_neurocore/accel/mojo/kernels/poisson.mojo"
    output = tmp_path / "libpoisson.so"
    subprocess.run(
        ["mojo", "build", "--emit", "shared-lib", "-o", str(output), str(source)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    library = ctypes.CDLL(str(output))
    assert library.poisson_simulate_c is not None


def test_staged_go_and_mojo_artifacts_reload_through_dispatcher() -> None:
    """Resolve actual staged libraries through the production loaders."""
    assert backends.ensure_go_loaded()
    assert backends.ensure_mojo_loaded()
    assert backends._go_lib is not None
    assert backends._mojo_lib is not None


def test_engine_import_failure_disables_only_the_optional_rust_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Module initialisation keeps the Python floor when PyO3 is unavailable."""
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
    """Every Julia discovery failure is a non-fatal unavailable result."""
    monkeypatch.setattr(backends, "_julia_module", None)
    monkeypatch.setattr(importlib_util, "find_spec", lambda _name: None)
    assert backends.ensure_julia_loaded() is False

    monkeypatch.setattr(importlib_util, "find_spec", lambda _name: object())
    monkeypatch.setattr(os.path, "isfile", lambda _path: False)
    assert backends.ensure_julia_loaded() is False

    monkeypatch.setattr(os.path, "isfile", lambda _path: True)

    def fail_import(_name: str) -> ModuleType:
        raise RuntimeError("Julia startup failed")

    monkeypatch.setattr(
        importlib,
        "import_module",
        fail_import,
    )
    assert backends.ensure_julia_loaded() is False

    sentinel = object()
    monkeypatch.setattr(backends, "_julia_module", sentinel)
    assert backends.ensure_julia_loaded() is True


@pytest.mark.parametrize(
    ("library_attribute", "flag_attribute", "loader"),
    [
        ("_go_lib", "_HAS_GO", backends.ensure_go_loaded),
        ("_mojo_lib", "_HAS_MOJO", backends.ensure_mojo_loaded),
    ],
)
def test_c_library_loaders_fail_closed_without_a_valid_abi(
    library_attribute: str,
    flag_attribute: str,
    loader: Callable[[], bool],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing files, loader errors, and absent symbols cannot claim readiness."""
    monkeypatch.setattr(backends, library_attribute, None)
    monkeypatch.setattr(backends, flag_attribute, False)
    monkeypatch.setattr(os.path, "isfile", lambda _path: False)
    assert loader() is False

    monkeypatch.setattr(os.path, "isfile", lambda _path: True)

    def fail_load(_path: str) -> object:
        raise OSError("invalid library")

    monkeypatch.setattr(
        ctypes,
        "CDLL",
        fail_load,
    )
    assert loader() is False

    monkeypatch.setattr(ctypes, "CDLL", lambda _path: object())
    assert loader() is False

    sentinel = object()
    monkeypatch.setattr(backends, library_attribute, sentinel)
    assert loader() is True
