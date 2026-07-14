# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real sigmoid-rate backend build and loading contracts

"""Build and resolve every maintained sigmoid-rate execution surface."""

from __future__ import annotations

import ctypes
import importlib
from pathlib import Path
import subprocess

import numpy as np

from sc_neurocore.accel import sigmoid_rate as backends

_REPOSITORY = Path(__file__).resolve().parents[1]
_GO_ROOT = _REPOSITORY / "src/sc_neurocore/accel/go"
_GO_LIBRARY = _GO_ROOT / "neurons/sigmoid_rate/libsigmoid_rate.so"
_GO_HEADER = _GO_ROOT / "neurons/sigmoid_rate/libsigmoid_rate.h"
_MOJO_SOURCE = _REPOSITORY / "src/sc_neurocore/accel/mojo/kernels/sigmoid_rate.mojo"
_MOJO_LIBRARY = _MOJO_SOURCE.with_name("libsigmoid_rate.so")


def _build_go() -> None:
    subprocess.run(
        [
            "go",
            "build",
            "-buildmode=c-shared",
            "-o",
            "neurons/sigmoid_rate/libsigmoid_rate.so",
            "./neurons/sigmoid_rate",
        ],
        cwd=_GO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _build_mojo() -> None:
    subprocess.run(
        [
            "mojo",
            "build",
            "--emit",
            "shared-lib",
            "-o",
            str(_MOJO_LIBRARY),
            str(_MOJO_SOURCE),
        ],
        cwd=_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_rust_engine_package_reexports_configurable_batch() -> None:
    """Reach the batch function through the installed production package."""
    engine = importlib.import_module("sc_neurocore_engine")
    assert callable(engine.py_sigmoid_rate_simulate)
    trace, final_rate = engine.py_sigmoid_rate_simulate(0.25, 10.0, 2.0, 1.0, 0.5, 6, 3.0)
    assert np.asarray(trace).shape == (6,)
    assert final_rate == np.asarray(trace)[-1]


def test_go_c_shared_build_exports_atomic_batch() -> None:
    """Build the real Go service ABI and execute its configured contract."""
    _build_go()
    assert _GO_LIBRARY.is_file()
    assert _GO_HEADER.is_file()
    assert "sigmoid_rate_simulate_c" in _GO_HEADER.read_text(encoding="utf-8")
    backends._go_lib = None
    backends._HAS_GO = False
    assert backends.ensure_go_loaded()
    trace, final_rate = backends.simulate_go(0.25, 10.0, 2.0, 1.0, 0.5, 6, 3.0)
    assert trace.shape == (6,)
    assert final_rate == trace[-1]


def test_mojo_shared_build_exports_callable_batch() -> None:
    """Build a genuine Mojo shared library with the required public symbol."""
    _build_mojo()
    assert _MOJO_LIBRARY.is_file()
    library = ctypes.CDLL(str(_MOJO_LIBRARY))
    assert callable(library.sigmoid_rate_simulate_c)
    backends._mojo_lib = None
    backends._HAS_MOJO = False
    assert backends.ensure_mojo_loaded()
    trace, final_rate = backends.simulate_mojo(0.25, 10.0, 2.0, 1.0, 0.5, 6, 3.0)
    assert trace.shape == (6,)
    assert final_rate == trace[-1]


def test_julia_module_exposes_configurable_batch() -> None:
    """Load and execute the real Julia implementation through JuliaCall."""
    assert backends.ensure_julia_loaded()
    trace, final_rate = backends.simulate_julia(0.25, 10.0, 2.0, 1.0, 0.5, 6, 3.0)
    assert trace.shape == (6,)
    assert final_rate == trace[-1]
