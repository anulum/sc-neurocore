# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive-threshold IF Julia facade

"""Dedicated Python facade for the maintained Julia adaptive-threshold kernel."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ._runtime import JULIA_MAIN as _jl
from ._runtime import KERNEL_DIR as _KERNEL_DIR
from ._runtime import is_julia_error

_LOADED = False


def _ensure_loaded() -> Any:
    """Include the maintained adaptive-threshold kernel on first use."""
    global _LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _LOADED:
        jl_path = _KERNEL_DIR / "adaptive_threshold_if.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"adaptive_threshold_if.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _LOADED = True
    return _jl.AdaptiveThresholdIFAccel


def _as_input(current: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Convert adaptive-threshold drive into a finite one-dimensional vector."""
    drive = np.ascontiguousarray(current, dtype=np.float64)
    if drive.ndim != 1:
        raise ValueError(f"current must be one-dimensional: got shape {drive.shape}")
    if not np.isfinite(drive).all():
        raise ValueError("current must contain only finite values")
    return drive


def simulate_adaptive_threshold_if(
    v_init: float,
    theta_init: float,
    v_rest: float,
    v_reset: float,
    theta_rest: float,
    delta_theta: float,
    tau_m: float,
    tau_theta: float,
    dt: float,
    current: npt.ArrayLike,
) -> dict[str, npt.NDArray[np.float64] | float | int]:
    """Run the Julia exact-relaxation recurrence with typed failure translation."""
    drive = _as_input(current)
    module = _ensure_loaded()
    v_out = np.empty(drive.size, dtype=np.float64)
    theta_out = np.empty(drive.size, dtype=np.float64)
    spikes_out = np.empty(drive.size, dtype=np.float64)
    try:
        v_final, theta_final, spike_count = module.simulate_adaptive_threshold_if_b(
            v_init,
            theta_init,
            v_rest,
            v_reset,
            theta_rest,
            delta_theta,
            tau_m,
            tau_theta,
            dt,
            drive,
            v_out,
            theta_out,
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
        "theta": theta_out,
        "spikes": spikes_out,
        "v_final": float(v_final),
        "theta_final": float(theta_final),
        "spike_count": int(spike_count),
    }
