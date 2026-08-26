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

_NON_RESETTING_LIF_LOADED = False
_SC_NON_RESETTING_ADAPTIVE_LIF_LOADED = False


def _ensure_non_resetting_lif_loaded() -> Any:
    """Include the source MAT(1) module into Julia Main on first use."""
    global _NON_RESETTING_LIF_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _NON_RESETTING_LIF_LOADED:
        jl_path = _KERNEL_DIR / "non_resetting_lif.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"non_resetting_lif.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _NON_RESETTING_LIF_LOADED = True
    return _jl.NonResettingLifAccel


def simulate_non_resetting_lif(
    currents: npt.ArrayLike,
    *,
    v: float = 0.0,
    theta: float = 0.0,
    refractory_remaining: float = 0.0,
    omega: float = 19.0,
    tau_m: float = 5.0,
    tau_theta: float = 50.0,
    alpha: float = 37.0,
    resistance: float = 50.0,
    refractory_period: float = 2.0,
    dt: float = 0.001,
) -> dict[str, object]:
    """Run a complete source MAT(1) state/event trace in Julia."""
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    if drive.ndim != 1 or not np.isfinite(drive).all():
        raise ValueError("Julia MAT(1) current must be finite and one-dimensional")
    module = _ensure_non_resetting_lif_loaded()
    state = module.NonResettingLIFNeuronState(
        v,
        theta,
        refractory_remaining,
        omega,
        tau_m,
        tau_theta,
        alpha,
        resistance,
        refractory_period,
        dt,
    )
    result = module.simulate(drive, state=state)
    return {
        "voltages": np.asarray(result.voltages, dtype=np.float64),
        "theta": np.asarray(result.theta, dtype=np.float64),
        "refractory": np.asarray(result.refractory, dtype=np.float64),
        "events": np.asarray(result.events, dtype=np.int64),
        "v_final": float(result.state.v),
        "theta_final": float(result.state.theta),
        "refractory_final": float(result.state.refractory_remaining),
    }


def _ensure_sc_non_resetting_adaptive_lif_loaded() -> Any:
    """Include the retained project module into Julia Main on first use."""
    global _SC_NON_RESETTING_ADAPTIVE_LIF_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _SC_NON_RESETTING_ADAPTIVE_LIF_LOADED:
        jl_path = _KERNEL_DIR / "sc_non_resetting_adaptive_lif.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"sc_non_resetting_adaptive_lif.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _SC_NON_RESETTING_ADAPTIVE_LIF_LOADED = True
    return _jl.SCNonResettingAdaptiveLifAccel


def simulate_sc_non_resetting_adaptive_lif(
    currents: npt.ArrayLike,
    *,
    v: float = -65.0,
    theta: float = -50.0,
    v_rest: float = -65.0,
    theta_rest: float = -50.0,
    delta_theta: float = 5.0,
    tau_m: float = 10.0,
    tau_theta: float = 50.0,
    r_m: float = 1.0,
    dt: float = 0.1,
) -> dict[str, object]:
    """Run a complete retained-project state/event trace in Julia."""
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    if drive.ndim != 1 or not np.isfinite(drive).all():
        raise ValueError("Julia SC adaptive LIF current must be finite and one-dimensional")
    module = _ensure_sc_non_resetting_adaptive_lif_loaded()
    state = module.SCNonResettingAdaptiveLIFNeuronState(
        v,
        theta,
        v_rest,
        theta_rest,
        delta_theta,
        tau_m,
        tau_theta,
        r_m,
        dt,
    )
    result = module.simulate(drive, state=state)
    return {
        "voltages": np.asarray(result.voltages, dtype=np.float64),
        "theta": np.asarray(result.theta, dtype=np.float64),
        "events": np.asarray(result.events, dtype=np.int64),
        "v_final": float(result.state.v),
        "theta_final": float(result.state.theta),
    }
