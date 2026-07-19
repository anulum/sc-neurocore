# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Go-backed adaptive-threshold exact-relaxation batch

"""Typed ctypes facade for the Go adaptive-threshold shared library."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import numpy.typing as npt

from sc_neurocore.neurons.models.adaptive_threshold_if import AdaptiveThresholdIFResult

_ACCEL_ROOT = Path(__file__).resolve().parents[2]
_LIB_PATH = _ACCEL_ROOT / "go" / "adaptive_threshold_if" / "libadaptive_threshold_if.so"
_MAX_NATIVE_STEPS = (1 << 31) - 1


def _configure_library(library: ctypes.CDLL) -> ctypes.CDLL:
    library.adaptive_threshold_if_simulate_c.argtypes = [
        ctypes.c_int32,
        *([ctypes.c_double] * 9),
        *([ctypes.c_void_p] * 7),
    ]
    library.adaptive_threshold_if_simulate_c.restype = ctypes.c_int32
    return library


def _load_library() -> tuple[ctypes.CDLL | None, bool]:
    try:
        return _configure_library(ctypes.CDLL(str(_LIB_PATH))), True
    except OSError:
        return None, False


_lib, _HAS_GO_ADAPTIVE_THRESHOLD_IF = _load_library()


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
) -> AdaptiveThresholdIFResult:
    """Run the Go implementation of the complete exact-relaxation recurrence."""
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
            f"libadaptive_threshold_if.so not built. Run: cd {_LIB_PATH.parent} && "
            "GOTOOLCHAIN=auto CGO_ENABLED=1 "
            f"go build -buildmode=c-shared -o {_LIB_PATH.name} ."
        )
    v_trace: npt.NDArray[np.float64] = np.empty(drive.size, dtype=np.float64)
    theta_trace: npt.NDArray[np.float64] = np.empty(drive.size, dtype=np.float64)
    spikes: npt.NDArray[np.float64] = np.empty(drive.size, dtype=np.float64)
    v_final: npt.NDArray[np.float64] = np.empty(1, dtype=np.float64)
    theta_final: npt.NDArray[np.float64] = np.empty(1, dtype=np.float64)
    spike_count: npt.NDArray[np.float64] = np.empty(1, dtype=np.float64)
    status = _lib.adaptive_threshold_if_simulate_c(
        ctypes.c_int32(drive.size),
        *(
            float(value)
            for value in (
                v_init,
                theta_init,
                v_rest,
                v_reset,
                theta_rest,
                delta_theta,
                tau_m,
                tau_theta,
                dt,
            )
        ),
        drive.ctypes.data,
        v_trace.ctypes.data,
        theta_trace.ctypes.data,
        spikes.ctypes.data,
        v_final.ctypes.data,
        theta_final.ctypes.data,
        spike_count.ctypes.data,
    )
    if status == 4:
        raise FloatingPointError(
            "adaptive_threshold_if_simulate_c rejected exact-relaxation candidate"
        )
    if status == 2:
        raise ValueError("adaptive_threshold_if_simulate_c rejected configuration")
    if status == 3:
        raise ValueError("adaptive_threshold_if_simulate_c rejected current")
    if status != 0:
        raise RuntimeError(f"adaptive_threshold_if_simulate_c rejected code {status}")
    return {
        "v": v_trace,
        "theta": theta_trace,
        "spikes": spikes,
        "v_final": float(v_final[0]),
        "theta_final": float(theta_final[0]),
        "spike_count": int(spike_count[0]),
    }


__all__ = ["_HAS_GO_ADAPTIVE_THRESHOLD_IF", "simulate_adaptive_threshold_if"]
