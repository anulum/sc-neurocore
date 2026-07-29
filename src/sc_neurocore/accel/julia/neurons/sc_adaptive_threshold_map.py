# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — dedicated Julia SC adaptive-map facade

"""Python facade for the maintained Julia SC adaptive-map kernel."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ._runtime import JULIA_MAIN as _jl
from ._runtime import KERNEL_DIR as _KERNEL_DIR
from ._runtime import is_julia_error

_LOADED = False


def _ensure_loaded() -> Any:
    """Include the project-model Julia kernel on first use."""
    global _LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _LOADED:
        path = _KERNEL_DIR / "sc_adaptive_threshold_map_neuron.jl"
        if not path.is_file():
            raise FileNotFoundError(f"SC adaptive-map Julia kernel missing at {path}")
        _jl.include(str(path))
        _LOADED = True
    return _jl.SCAdaptiveThresholdMapNeuronAccel


def simulate_sc_adaptive_threshold_map(
    x: float,
    theta: float,
    k: float,
    beta: float,
    gamma: float,
    theta_spike: float,
    x_threshold: float,
    current: npt.ArrayLike,
) -> dict[str, object]:
    """Run a checked Julia batch and translate typed numerical failures."""
    drive = np.ascontiguousarray(current, dtype=np.float64)
    if drive.ndim != 1:
        raise ValueError(f"current must be one-dimensional: got shape {drive.shape}")
    if not np.isfinite(drive).all():
        raise ValueError("current must contain only finite values")
    traces = [np.empty(drive.size, dtype=np.float64) for _ in range(3)]
    module = _ensure_loaded()
    try:
        x_final, theta_final, spike_count = module.simulate_sc_adaptive_threshold_map_b(
            x, theta, k, beta, gamma, theta_spike, x_threshold, drive, *traces
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
        "x": traces[0],
        "theta": traces[1],
        "spikes": traces[2],
        "x_final": float(x_final),
        "theta_final": float(theta_final),
        "spike_count": int(spike_count),
    }


__all__ = ["simulate_sc_adaptive_threshold_map"]
