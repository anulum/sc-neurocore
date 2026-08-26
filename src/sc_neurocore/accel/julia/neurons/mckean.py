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

_MCKEAN_LOADED = False
_SC_TRIANGULAR_MCKEAN_LOADED = False

def _ensure_mckean_loaded() -> Any:
    """Include the source-bound McKean module."""
    global _MCKEAN_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _MCKEAN_LOADED:
        jl_path = _KERNEL_DIR / "mckean.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"mckean.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _MCKEAN_LOADED = True
    return _jl.McKeanAccel


def simulate_mckean(
    currents: npt.ArrayLike,
    *,
    v: float = 0.0,
    w: float = 0.0,
    a: float = 0.25,
    lambda_: float = 1.0,
    mu: float = 1.0,
    b: float = 0.01,
    dt: float = 0.1,
) -> dict[str, object]:
    """Run a complete source McKean state/event trace in Julia."""
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    if drive.ndim != 1 or not np.isfinite(drive).all():
        raise ValueError("Julia McKean current must be finite and one-dimensional")
    module = _ensure_mckean_loaded()
    state = module.McKeanNeuronState(v, w, a, lambda_, mu, b, dt)
    result = module.simulate(drive, state=state)
    return {
        "voltages": np.asarray(result.voltages, dtype=np.float64),
        "recovery": np.asarray(result.recovery, dtype=np.float64),
        "events": np.asarray(result.events, dtype=np.int64),
        "v_final": float(result.state.v),
        "w_final": float(result.state.w),
    }


def _ensure_sc_triangular_mckean_loaded() -> Any:
    """Include the retained SC triangular recurrence module."""
    global _SC_TRIANGULAR_MCKEAN_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _SC_TRIANGULAR_MCKEAN_LOADED:
        jl_path = _KERNEL_DIR / "sc_triangular_mckean.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"sc_triangular_mckean.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _SC_TRIANGULAR_MCKEAN_LOADED = True
    return _jl.SCTriangularMcKeanAccel
