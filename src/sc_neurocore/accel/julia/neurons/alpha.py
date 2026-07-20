# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Alpha-synapse Julia facade

"""Dedicated Python facade for the maintained Julia alpha-synapse kernel."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ._runtime import JULIA_MAIN as _jl
from ._runtime import KERNEL_DIR as _KERNEL_DIR
from ._runtime import is_julia_error

_LOADED = False


def _ensure_loaded() -> Any:
    """Include the maintained alpha-synapse kernel on first use."""
    global _LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _LOADED:
        jl_path = _KERNEL_DIR / "alpha.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"alpha.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _LOADED = True
    return _jl.AlphaAccel


def _as_input(current: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert one alpha drive into a finite one-dimensional vector."""
    drive = np.ascontiguousarray(current, dtype=np.float64)
    if drive.ndim != 1:
        raise ValueError(f"current must be one-dimensional: got shape {drive.shape}")
    if not np.isfinite(drive).all():
        raise ValueError("current must contain only finite values")
    return drive


def simulate_alpha(
    v_init: float,
    a_exc_init: float,
    i_exc_init: float,
    a_inh_init: float,
    i_inh_init: float,
    v_rest: float,
    v_threshold: float,
    tau_v: float,
    tau_exc: float,
    tau_inh: float,
    dt: float,
    exc_current: npt.ArrayLike,
    inh_current: npt.ArrayLike,
) -> dict[str, npt.NDArray[np.float64] | float | int]:
    """Run the Julia exact-flow recurrence with typed failure translation."""
    exc_drive = _as_input(exc_current)
    inh_drive = _as_input(inh_current)
    module = _ensure_loaded()
    size = exc_drive.size
    if inh_drive.size != size:
        raise ValueError("inh_current length mismatch")
    v_out = np.empty(size, dtype=np.float64)
    a_exc_out = np.empty(size, dtype=np.float64)
    i_exc_out = np.empty(size, dtype=np.float64)
    a_inh_out = np.empty(size, dtype=np.float64)
    i_inh_out = np.empty(size, dtype=np.float64)
    spikes_out = np.empty(size, dtype=np.float64)
    try:
        (
            v_final,
            a_exc_final,
            i_exc_final,
            a_inh_final,
            i_inh_final,
            spike_count,
        ) = module.simulate_alpha_b(
            v_init,
            a_exc_init,
            i_exc_init,
            a_inh_init,
            i_inh_init,
            v_rest,
            v_threshold,
            tau_v,
            tau_exc,
            tau_inh,
            dt,
            exc_drive,
            inh_drive,
            v_out,
            a_exc_out,
            i_exc_out,
            a_inh_out,
            i_inh_out,
            spikes_out,
        )
    except Exception as exc:
        if not is_julia_error(exc):
            raise
        julia_exception = getattr(exc, "exception", None)
        if module.is_configuration_error(julia_exception):
            raise ValueError(str(exc)) from exc
        if module.is_candidate_error(julia_exception):
            raise FloatingPointError(str(exc)) from exc
        raise
    return {
        "v": v_out,
        "a_exc": a_exc_out,
        "i_exc": i_exc_out,
        "a_inh": a_inh_out,
        "i_inh": i_inh_out,
        "spikes": spikes_out,
        "v_final": float(v_final),
        "a_exc_final": float(a_exc_final),
        "i_exc_final": float(i_exc_final),
        "a_inh_final": float(a_inh_final),
        "i_inh_final": float(i_inh_final),
        "spike_count": int(spike_count),
    }
