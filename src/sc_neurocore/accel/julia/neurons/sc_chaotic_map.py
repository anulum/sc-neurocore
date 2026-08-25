# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — dedicated Julia facade for the SC two-state chaotic map

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ._runtime import JULIA_MAIN as _jl
from ._runtime import KERNEL_DIR as _KERNEL_DIR
from ._runtime import is_julia_error

_LOADED = False


def _ensure_loaded() -> Any:
    global _LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _LOADED:
        path = _KERNEL_DIR / "sc_chaotic_map_neuron.jl"
        if not path.is_file():
            raise FileNotFoundError(f"SC chaotic-map Julia kernel missing at {path}")
        _jl.include(str(path))
        _LOADED = True
    return _jl.SCChaoticMapNeuronAccel


def simulate_sc_chaotic_map(
    x: float,
    y: float,
    k_f: float,
    k_s: float,
    alpha: float,
    delta: float,
    x_threshold: float,
    current: npt.ArrayLike,
) -> dict[str, object]:
    drive = np.ascontiguousarray(current, dtype=np.float64)
    if drive.ndim != 1 or not np.isfinite(drive).all():
        raise ValueError("current must be a finite one-dimensional array")
    module = _ensure_loaded()
    try:
        result = module.simulate_sc_chaotic_map(x, y, k_f, k_s, alpha, delta, x_threshold, drive)
    except Exception as exc:
        if is_julia_error(exc):
            raise ValueError(str(exc)) from exc
        raise
    return {
        "x": np.asarray(result[0], dtype=np.float64),
        "y": np.asarray(result[1], dtype=np.float64),
        "spikes": np.asarray(result[2], dtype=np.float64),
        "x_final": float(result[3]),
        "y_final": float(result[4]),
        "spike_count": int(result[5]),
    }


__all__ = ["simulate_sc_chaotic_map"]
