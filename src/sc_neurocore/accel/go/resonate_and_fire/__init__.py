# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Go-backed resonate-and-fire exact-flow batch

"""Typed ctypes facade for the Go resonate-and-fire shared library."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import numpy.typing as npt

from sc_neurocore.neurons.models.resonate_and_fire import ResonateAndFireResult

_ACCEL_ROOT = Path(__file__).resolve().parents[2]
_LIB_PATH = _ACCEL_ROOT / "go" / "resonate_and_fire" / "libresonate_and_fire.so"
_MAX_NATIVE_STEPS = (1 << 31) - 1


def _configure_library(library: ctypes.CDLL) -> ctypes.CDLL:
    library.resonate_and_fire_simulate_c.argtypes = [
        ctypes.c_int32,
        *([ctypes.c_double] * 6),
        *([ctypes.c_void_p] * 7),
    ]
    library.resonate_and_fire_simulate_c.restype = ctypes.c_int32
    return library


def _load_library() -> tuple[ctypes.CDLL | None, bool]:
    try:
        return _configure_library(ctypes.CDLL(str(_LIB_PATH))), True
    except OSError:
        return None, False


_lib, _HAS_GO_RESONATE_AND_FIRE = _load_library()


def simulate_resonate_and_fire(
    x_init: float,
    y_init: float,
    b: float,
    omega: float,
    threshold: float,
    dt: float,
    current: npt.ArrayLike,
) -> ResonateAndFireResult:
    """Run the Go implementation of the complete exact-flow recurrence."""
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
            f"libresonate_and_fire.so not built. Run: cd {_LIB_PATH.parent} && "
            "GOTOOLCHAIN=auto CGO_ENABLED=1 "
            f"go build -buildmode=c-shared -o {_LIB_PATH.name} ."
        )
    x_trace: npt.NDArray[np.float64] = np.empty(drive.size, dtype=np.float64)
    y_trace: npt.NDArray[np.float64] = np.empty(drive.size, dtype=np.float64)
    spikes: npt.NDArray[np.float64] = np.empty(drive.size, dtype=np.float64)
    x_final: npt.NDArray[np.float64] = np.empty(1, dtype=np.float64)
    y_final: npt.NDArray[np.float64] = np.empty(1, dtype=np.float64)
    spike_count: npt.NDArray[np.float64] = np.empty(1, dtype=np.float64)
    status = _lib.resonate_and_fire_simulate_c(
        ctypes.c_int32(drive.size),
        *(float(value) for value in (x_init, y_init, b, omega, threshold, dt)),
        drive.ctypes.data,
        x_trace.ctypes.data,
        y_trace.ctypes.data,
        spikes.ctypes.data,
        x_final.ctypes.data,
        y_final.ctypes.data,
        spike_count.ctypes.data,
    )
    if status == 4:
        raise FloatingPointError("resonate_and_fire_simulate_c rejected exact-flow candidate")
    if status == 2:
        raise ValueError("resonate_and_fire_simulate_c rejected configuration")
    if status == 3:
        raise ValueError("resonate_and_fire_simulate_c rejected current")
    if status != 0:
        raise RuntimeError(f"resonate_and_fire_simulate_c rejected code {status}")
    return {
        "x": x_trace,
        "y": y_trace,
        "spikes": spikes,
        "x_final": float(x_final[0]),
        "y_final": float(y_final[0]),
        "spike_count": int(spike_count[0]),
    }


__all__ = ["_HAS_GO_RESONATE_AND_FIRE", "simulate_resonate_and_fire"]
