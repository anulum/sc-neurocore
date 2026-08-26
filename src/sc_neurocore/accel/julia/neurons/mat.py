# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — dedicated Julia neuron facade

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ._runtime import JULIA_MAIN as _jl
from ._runtime import KERNEL_DIR as _KERNEL_DIR

_MAT_LOADED = False
_SC_RESETTING_MAT_LOADED = False

def _ensure_mat_loaded() -> Any:
    """Include the source MAT* module into Julia Main on first use."""
    global _MAT_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _MAT_LOADED:
        jl_path = _KERNEL_DIR / "mat.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"mat.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _MAT_LOADED = True
    return _jl.MatAccel


def simulate_mat(
    currents: npt.ArrayLike,
    *,
    v: float = 0.0,
    theta1: float = 0.0,
    theta2: float = 0.0,
    refractory_remaining: float = 0.0,
    omega: float = 19.0,
    tau_m: float = 5.0,
    tau_1: float = 10.0,
    tau_2: float = 200.0,
    alpha_1: float = 37.0,
    alpha_2: float = 2.0,
    resistance: float = 50.0,
    refractory_period: float = 2.0,
    dt: float = 0.001,
) -> dict[str, object]:
    """Run a complete non-resetting MAT* state/event trace in Julia."""
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    if drive.ndim != 1 or not np.isfinite(drive).all():
        raise ValueError("Julia MAT current must be a finite one-dimensional array")
    module = _ensure_mat_loaded()
    state = module.MATNeuronState(
        v,
        theta1,
        theta2,
        refractory_remaining,
        omega,
        tau_m,
        tau_1,
        tau_2,
        alpha_1,
        alpha_2,
        resistance,
        refractory_period,
        dt,
    )
    result = module.simulate(drive, state=state)
    return {
        "voltages": np.asarray(result.voltages, dtype=np.float64),
        "theta1": np.asarray(result.theta1, dtype=np.float64),
        "theta2": np.asarray(result.theta2, dtype=np.float64),
        "refractory": np.asarray(result.refractory, dtype=np.float64),
        "events": np.asarray(result.events, dtype=np.int64),
        "v_final": float(result.state.v),
        "theta1_final": float(result.state.theta1),
        "theta2_final": float(result.state.theta2),
        "refractory_final": float(result.state.refractory_remaining),
    }


def _ensure_sc_resetting_mat_loaded() -> Any:
    """Include the project resetting-MAT module into Julia Main on first use."""
    global _SC_RESETTING_MAT_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _SC_RESETTING_MAT_LOADED:
        jl_path = _KERNEL_DIR / "sc_resetting_mat.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"sc_resetting_mat.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _SC_RESETTING_MAT_LOADED = True
    return _jl.SCResettingMatAccel


def simulate_sc_resetting_mat(
    currents: npt.ArrayLike,
    *,
    v: float = -70.0,
    theta1: float = 0.0,
    theta2: float = 0.0,
    v_rest: float = -70.0,
    v_reset: float = -70.0,
    v_threshold_base: float = -50.0,
    tau_m: float = 10.0,
    tau_1: float = 10.0,
    tau_2: float = 200.0,
    h1: float = 5.0,
    h2: float = 3.0,
    resistance: float = 1.0,
    dt: float = 1.0,
) -> dict[str, object]:
    """Run a complete candidate-first RK4/reset trace in Julia."""
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    if drive.ndim != 1 or not np.isfinite(drive).all():
        raise ValueError("Julia SC resetting-MAT current must be finite and one-dimensional")
    module = _ensure_sc_resetting_mat_loaded()
    state = module.SCResettingMATNeuronState(
        v,
        theta1,
        theta2,
        v_rest,
        v_reset,
        v_threshold_base,
        tau_m,
        tau_1,
        tau_2,
        h1,
        h2,
        resistance,
        dt,
    )
    result = module.simulate(drive, state=state)
    return {
        "voltages": np.asarray(result.voltages, dtype=np.float64),
        "theta1": np.asarray(result.theta1, dtype=np.float64),
        "theta2": np.asarray(result.theta2, dtype=np.float64),
        "events": np.asarray(result.events, dtype=np.int64),
        "v_final": float(result.state.v),
        "theta1_final": float(result.state.theta1),
        "theta2_final": float(result.state.theta2),
    }
