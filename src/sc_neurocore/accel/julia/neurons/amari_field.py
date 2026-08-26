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

_AMARI_FIELD_LOADED = False

def _ensure_amari_field_loaded() -> Any:
    """Include ``amari_field.jl`` into Julia Main on first use."""
    global _AMARI_FIELD_LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _AMARI_FIELD_LOADED:
        jl_path = _KERNEL_DIR / "amari_field.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"amari_field.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _AMARI_FIELD_LOADED = True
    return _jl.AmariFieldAccel


def simulate_amari_field(
    u_init: npt.ArrayLike,
    tau: float,
    a_exc: float,
    a_width: float,
    b_inh: float,
    b_width: float,
    dx: float,
    dt: float,
    currents: npt.ArrayLike,
) -> dict[str, object]:
    """Run the complete vector batch through the native Julia module."""
    state = np.ascontiguousarray(u_init, dtype=np.float64)
    drive = np.ascontiguousarray(currents, dtype=np.float64)
    if state.ndim != 1 or drive.ndim != 2 or drive.shape[1] != state.size:
        raise ValueError("Julia Amari state/input shape mismatch")
    if not np.isfinite(state).all() or not np.isfinite(drive).all():
        raise ValueError("Julia Amari state and input must be finite")
    states = np.empty_like(drive)
    rates = np.empty(drive.shape[0], dtype=np.float64)
    module = _ensure_amari_field_loaded()
    final = np.asarray(
        module.simulate_amari_field_b(
            state,
            tau,
            a_exc,
            a_width,
            b_inh,
            b_width,
            dx,
            dt,
            drive,
            states,
            rates,
        ),
        dtype=np.float64,
    )
    return {"states": states, "mean_rates": rates, "final_state": final}
