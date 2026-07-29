# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Go-backed source-faithful Aihara batch

"""Typed ctypes facade for the Go Aihara shared library."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import numpy.typing as npt

from sc_neurocore.neurons.models.aihara_map_neuron import AiharaMapResult

_ACCEL_ROOT = Path(__file__).resolve().parents[2]
_LIB_PATH = _ACCEL_ROOT / "go" / "aihara_map" / "libaihara_map.so"
_MAX_NATIVE_STEPS = (1 << 31) - 1


def _configure(library: ctypes.CDLL) -> ctypes.CDLL:
    library.aihara_map_simulate_c.argtypes = [
        ctypes.c_int32,
        *([ctypes.c_double] * 5),
        *([ctypes.c_void_p] * 7),
    ]
    library.aihara_map_simulate_c.restype = ctypes.c_int32
    return library


try:
    _lib: ctypes.CDLL | None = _configure(ctypes.CDLL(str(_LIB_PATH)))
except OSError:
    _lib = None
_HAS_GO_AIHARA_MAP = _lib is not None


def simulate_aihara_map(
    y: float,
    k: float,
    alpha: float,
    bias: float,
    epsilon: float,
    current: npt.ArrayLike,
) -> AiharaMapResult:
    """Run the checked Go implementation of Aihara Eqs. 10-12."""
    logical = np.asarray(current)
    if logical.ndim != 1:
        raise ValueError(f"current must be one-dimensional: got shape {logical.shape}")
    if logical.size > _MAX_NATIVE_STEPS:
        raise ValueError(f"current exceeds the signed-32-bit step limit: {logical.size}")
    drive = np.ascontiguousarray(logical, dtype=np.float64)
    if not np.isfinite(drive).all():
        raise ValueError("current must contain only finite values")
    if _lib is None:
        raise ImportError(
            f"libaihara_map.so not built. Run: cd {_LIB_PATH.parent} && "
            "GOTOOLCHAIN=auto CGO_ENABLED=1 go build -buildmode=c-shared "
            f"-o {_LIB_PATH.name} ."
        )
    traces = [np.empty(drive.size, dtype=np.float64) for _ in range(3)]
    finals = [np.empty(1, dtype=np.float64) for _ in range(3)]
    status = _lib.aihara_map_simulate_c(
        ctypes.c_int32(drive.size),
        *(float(value) for value in (y, k, alpha, bias, epsilon)),
        drive.ctypes.data,
        *(array.ctypes.data for array in traces),
        *(array.ctypes.data for array in finals),
    )
    if status == 2:
        raise ValueError("aihara_map_simulate_c rejected configuration")
    if status == 3:
        raise ValueError("aihara_map_simulate_c rejected current")
    if status == 4:
        raise FloatingPointError("aihara_map_simulate_c rejected map candidate")
    if status != 0:
        raise RuntimeError(f"aihara_map_simulate_c rejected code {status}")
    return {
        "y": traces[0],
        "x": traces[1],
        "spikes": traces[2],
        "y_final": float(finals[0][0]),
        "x_final": float(finals[1][0]),
        "spike_count": int(finals[2][0]),
    }


__all__ = ["_HAS_GO_AIHARA_MAP", "simulate_aihara_map"]
