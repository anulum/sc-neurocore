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

_ENERGY_LIF_LOADED = False
_SC_NORMALIZED_ENERGY_LIF_LOADED = False

def _ensure_energy_lif_loaded() -> Any:
    """Include the source-faithful Fardet-Levina eLIF module."""
    global _ENERGY_LIF_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _ENERGY_LIF_LOADED:
        jl_path = _KERNEL_DIR / "energy_lif.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"energy_lif.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _ENERGY_LIF_LOADED = True
    return _jl.EnergyLifAccel


def simulate_energy_lif(
    currents: npt.ArrayLike,
    *,
    v: float = -61.0,
    epsilon: float = 0.32,
    capacitance: float = 100.0,
    g_leak: float = 9.0,
    e_0: float = -62.5,
    e_u: float = -58.5,
    e_d: float = -40.0,
    e_f: float = -62.0,
    v_threshold: float = -59.0,
    v_reset: float = -62.0,
    alpha: float = 1.0,
    epsilon_0: float = 0.5,
    epsilon_c: float = 0.18,
    delta: float = 0.01,
    tau_e: float = 200.0,
    dt: float = 0.1,
) -> dict[str, object]:
    """Run the complete source-faithful eLIF trace in Julia."""
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    if drive.ndim != 1 or not np.isfinite(drive).all():
        raise ValueError("Julia EnergyLIF current must be finite and one-dimensional")
    module = _ensure_energy_lif_loaded()
    state = module.EnergyLIFNeuronState(
        v,
        epsilon,
        capacitance,
        g_leak,
        e_0,
        e_u,
        e_d,
        e_f,
        v_threshold,
        v_reset,
        alpha,
        epsilon_0,
        epsilon_c,
        delta,
        tau_e,
        dt,
    )
    result = module.simulate(drive, state=state)
    return {
        "voltages": np.asarray(result.voltages, dtype=np.float64),
        "epsilon": np.asarray(result.epsilon, dtype=np.float64),
        "events": np.asarray(result.events, dtype=np.int64),
        "v_final": float(result.state.v),
        "epsilon_final": float(result.state.epsilon),
    }


def _ensure_sc_normalized_energy_lif_loaded() -> Any:
    """Include the retained normalized-energy SC module."""
    global _SC_NORMALIZED_ENERGY_LIF_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _SC_NORMALIZED_ENERGY_LIF_LOADED:
        jl_path = _KERNEL_DIR / "sc_normalized_energy_lif.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"sc_normalized_energy_lif.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _SC_NORMALIZED_ENERGY_LIF_LOADED = True
    return _jl.SCNormalizedEnergyLifAccel


def simulate_sc_normalized_energy_lif(
    currents: npt.ArrayLike,
    *,
    v: float = -70.0,
    epsilon: float = 1.0,
    v_rest: float = -70.0,
    v_reset: float = -70.0,
    v_threshold: float = -50.0,
    tau_m: float = 10.0,
    tau_e: float = 500.0,
    alpha: float = 0.1,
    epsilon_0: float = 1.0,
    resistance: float = 1.0,
    dt: float = 1.0,
) -> dict[str, object]:
    """Run the complete retained normalized-energy SC trace in Julia."""
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    if drive.ndim != 1 or not np.isfinite(drive).all():
        raise ValueError("Julia SC EnergyLIF current must be finite and one-dimensional")
    module = _ensure_sc_normalized_energy_lif_loaded()
    state = module.SCNormalizedEnergyLIFNeuronState(
        v,
        epsilon,
        v_rest,
        v_reset,
        v_threshold,
        tau_m,
        tau_e,
        alpha,
        epsilon_0,
        resistance,
        dt,
    )
    result = module.simulate(drive, state=state)
    return {
        "voltages": np.asarray(result.voltages, dtype=np.float64),
        "epsilon": np.asarray(result.epsilon, dtype=np.float64),
        "events": np.asarray(result.events, dtype=np.int64),
        "v_final": float(result.state.v),
        "epsilon_final": float(result.state.epsilon),
    }
