# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo-backed SC two-state chaotic map

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import numpy.typing as npt

from sc_neurocore.neurons.models.sc_chaotic_map_neuron import SCChaoticMapResult

_ACCEL_ROOT = Path(__file__).resolve().parents[2]
_LIB_PATH = _ACCEL_ROOT / "mojo" / "sc_chaotic_map" / "libsc_chaotic_map.so"
_MAX_NATIVE_STEPS = (1 << 31) - 1


def _configure(library: ctypes.CDLL) -> ctypes.CDLL:
    library.sc_chaotic_map_simulate_c.argtypes = [
        ctypes.c_int32, *([ctypes.c_double] * 7), *([ctypes.c_void_p] * 7)
    ]
    library.sc_chaotic_map_simulate_c.restype = ctypes.c_int32
    return library


try:
    _lib: ctypes.CDLL | None = _configure(ctypes.CDLL(str(_LIB_PATH)))
except OSError:
    _lib = None
_HAS_MOJO_SC_CHAOTIC_MAP = _lib is not None


def simulate_sc_chaotic_map(
    x: float, y: float, k_f: float, k_s: float, alpha: float, delta: float,
    x_threshold: float, current: npt.ArrayLike,
) -> SCChaoticMapResult:
    logical = np.asarray(current)
    if logical.ndim != 1:
        raise ValueError(f"current must be one-dimensional: got shape {logical.shape}")
    if logical.size > _MAX_NATIVE_STEPS:
        raise ValueError(f"current exceeds the signed-32-bit step limit: {logical.size}")
    drive = np.ascontiguousarray(logical, dtype=np.float64)
    if not np.isfinite(drive).all():
        raise ValueError("current must contain only finite values")
    if _lib is None:
        raise ImportError(f"libsc_chaotic_map.so not built under {_LIB_PATH.parent}")
    traces = [np.empty(drive.size, dtype=np.float64) for _ in range(3)]
    finals = [np.empty(1, dtype=np.float64) for _ in range(3)]
    status = _lib.sc_chaotic_map_simulate_c(
        ctypes.c_int32(drive.size),
        *(float(value) for value in (x, y, k_f, k_s, alpha, delta, x_threshold)),
        drive.ctypes.data,
        *(array.ctypes.data for array in traces),
        *(array.ctypes.data for array in finals),
    )
    if status in {2, 3}:
        raise ValueError(f"sc_chaotic_map_simulate_c rejected code {status}")
    if status == 4:
        raise FloatingPointError("sc_chaotic_map_simulate_c rejected map candidate")
    if status != 0:
        raise RuntimeError(f"sc_chaotic_map_simulate_c rejected code {status}")
    return {"x": traces[0], "y": traces[1], "spikes": traces[2],
            "x_final": float(finals[0][0]), "y_final": float(finals[1][0]),
            "spike_count": int(finals[2][0])}


__all__ = ["_HAS_MOJO_SC_CHAOTIC_MAP", "simulate_sc_chaotic_map"]
