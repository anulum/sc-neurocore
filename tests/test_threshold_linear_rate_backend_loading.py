# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real threshold-linear backend build and loading contracts

"""Build and resolve every maintained threshold-linear execution surface."""

from __future__ import annotations

import ctypes
import importlib
import subprocess
from pathlib import Path

import numpy as np

from sc_neurocore.accel import threshold_linear_rate as backends
from sc_neurocore.accel.mojo.isa_baseline import pin_isa

_REPOSITORY = Path(__file__).resolve().parents[1]
_GO_ROOT = _REPOSITORY / "src/sc_neurocore/accel/go"
_GO_LIBRARY = _GO_ROOT / "neurons/threshold_linear_rate/libthreshold_linear_rate.so"
_GO_HEADER = _GO_ROOT / "neurons/threshold_linear_rate/libthreshold_linear_rate.h"
_MOJO_SOURCE = _REPOSITORY / "src/sc_neurocore/accel/mojo/kernels/threshold_linear_rate.mojo"
_MOJO_LIBRARY = _MOJO_SOURCE.with_name("libthreshold_linear_rate.so")


def _build_go() -> None:
    subprocess.run(
        [
            "go",
            "build",
            "-buildmode=c-shared",
            "-o",
            "neurons/threshold_linear_rate/libthreshold_linear_rate.so",
            "./neurons/threshold_linear_rate",
        ],
        cwd=_GO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _build_mojo() -> None:
    subprocess.run(
        pin_isa(
            [
                "mojo",
                "build",
                "--emit",
                "shared-lib",
                "-o",
                str(_MOJO_LIBRARY),
                str(_MOJO_SOURCE),
            ]
        ),
        cwd=_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_rust_engine_package_reexports_configurable_batch() -> None:
    engine = importlib.import_module("sc_neurocore_engine")
    neuron = engine.ThresholdLinearRateNeuron(0.25, 1.5, 2.0)
    assert neuron.step(3.0) == 3.0
    assert neuron.get_state() == {"r": 3.0, "theta": 1.5, "gain": 2.0}
    neuron.reset()
    assert neuron.get_state() == {"r": 0.0, "theta": 1.5, "gain": 2.0}
    assert callable(engine.py_threshold_linear_rate_simulate)
    trace, final_rate = engine.py_threshold_linear_rate_simulate(0.25, 1.5, 2.0, 6, 3.0)
    np.testing.assert_array_equal(trace, np.full(6, 3.0))
    assert final_rate == 3.0


def test_go_c_shared_build_exports_atomic_batch() -> None:
    _build_go()
    assert _GO_LIBRARY.is_file()
    assert _GO_HEADER.is_file()
    assert "threshold_linear_rate_simulate_c" in _GO_HEADER.read_text(encoding="utf-8")
    backends._go_lib = None
    backends._HAS_GO = False
    assert backends.ensure_go_loaded()
    trace, final_rate = backends.simulate_go(0.25, 1.5, 2.0, 6, 3.0)
    np.testing.assert_array_equal(trace, np.full(6, 3.0))
    assert final_rate == 3.0


def test_mojo_shared_build_exports_callable_batch() -> None:
    _build_mojo()
    assert _MOJO_LIBRARY.is_file()
    library = ctypes.CDLL(str(_MOJO_LIBRARY))
    assert callable(library.threshold_linear_rate_simulate_c)
    backends._mojo_lib = None
    backends._HAS_MOJO = False
    assert backends.ensure_mojo_loaded()
    trace, final_rate = backends.simulate_mojo(0.25, 1.5, 2.0, 6, 3.0)
    np.testing.assert_array_equal(trace, np.full(6, 3.0))
    assert final_rate == 3.0


def test_julia_module_exposes_configurable_batch() -> None:
    assert backends.ensure_julia_loaded()
    trace, final_rate = backends.simulate_julia(0.25, 1.5, 2.0, 6, 3.0)
    np.testing.assert_array_equal(trace, np.full(6, 3.0))
    assert final_rate == 3.0
