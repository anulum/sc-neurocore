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

_SIGMA_DELTA_LOADED = False
_SC_SIGMA_DELTA_ACCUMULATOR_LOADED = False

def _ensure_sigma_delta_loaded() -> Any:
    """Include the sampled APSDM source module into Julia Main."""
    global _SIGMA_DELTA_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _SIGMA_DELTA_LOADED:
        jl_path = _KERNEL_DIR / "sigma_delta.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"sigma_delta.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _SIGMA_DELTA_LOADED = True
    return _jl.SigmaDeltaAccel


def simulate_sigma_delta(
    currents: npt.ArrayLike,
    *,
    sigma: float = 0.0,
    reconstruction: float = 0.0,
    delta: float = 1.0,
    tau_reconstruction: float = 10.0,
    dt: float = 0.1,
) -> dict[str, object]:
    """Run the complete sampled APSDM trace in Julia."""
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    if drive.ndim != 1 or not np.isfinite(drive).all():
        raise ValueError("Julia SigmaDelta current must be finite and one-dimensional")
    module = _ensure_sigma_delta_loaded()
    state = module.SigmaDeltaNeuronState(sigma, reconstruction, delta, tau_reconstruction, dt)
    result = module.simulate(drive, state=state)
    return {
        "sigma": np.asarray(result.sigma, dtype=np.float64),
        "reconstruction": np.asarray(result.reconstruction, dtype=np.float64),
        "events": np.asarray(result.events, dtype=np.int64),
        "sigma_final": float(result.state.sigma),
        "reconstruction_final": float(result.state.reconstruction),
    }


def _ensure_sc_sigma_delta_accumulator_loaded() -> Any:
    """Include the retained bipolar project module into Julia Main."""
    global _SC_SIGMA_DELTA_ACCUMULATOR_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _SC_SIGMA_DELTA_ACCUMULATOR_LOADED:
        jl_path = _KERNEL_DIR / "sc_sigma_delta_accumulator.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"sc_sigma_delta_accumulator.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _SC_SIGMA_DELTA_ACCUMULATOR_LOADED = True
    return _jl.SCSigmaDeltaAccumulatorAccel


def simulate_sc_sigma_delta_accumulator(
    currents: npt.ArrayLike,
    *,
    sigma: float = 0.0,
    v_threshold: float = 1.0,
) -> dict[str, object]:
    """Run the complete retained bipolar accumulator trace in Julia."""
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    if drive.ndim != 1 or not np.isfinite(drive).all():
        raise ValueError("Julia SC SigmaDelta current must be finite and one-dimensional")
    module = _ensure_sc_sigma_delta_accumulator_loaded()
    state = module.SCSigmaDeltaAccumulatorState(sigma, v_threshold)
    result = module.simulate(drive, state=state)
    return {
        "sigma": np.asarray(result.sigma, dtype=np.float64),
        "events": np.asarray(result.events, dtype=np.int64),
        "sigma_final": float(result.state.sigma),
    }
