# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo-backed alpha-synapse exact-flow batch

"""Typed ctypes facade for the Mojo alpha-synapse shared library."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import numpy.typing as npt

from sc_neurocore.neurons.models.alpha import AlphaResult

_ACCEL_ROOT = Path(__file__).resolve().parents[2]
_LIB_PATH = _ACCEL_ROOT / "mojo" / "alpha" / "libalpha.so"
_MAX_NATIVE_STEPS = (1 << 31) - 1


def _configure_library(library: ctypes.CDLL) -> ctypes.CDLL:
    library.alpha_simulate_c.argtypes = [
        ctypes.c_int32,
        *([ctypes.c_double] * 11),
        *([ctypes.c_void_p] * 14),
    ]
    library.alpha_simulate_c.restype = ctypes.c_int32
    return library


def _load_library() -> tuple[ctypes.CDLL | None, bool]:
    try:
        return _configure_library(ctypes.CDLL(str(_LIB_PATH))), True
    except OSError:
        return None, False


_lib, _HAS_MOJO_ALPHA = _load_library()


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
    inh_current: npt.ArrayLike | float = 0.0,
) -> AlphaResult:
    """Run the Mojo implementation of the complete exact-flow recurrence."""
    exc_logical = np.asarray(exc_current)
    if exc_logical.ndim != 1:
        raise ValueError(f"exc_current must be one-dimensional: got shape {exc_logical.shape}")
    if exc_logical.size > _MAX_NATIVE_STEPS:
        raise ValueError(f"exc_current exceeds the signed-32-bit step limit: {exc_logical.size}")
    exc_drive = np.ascontiguousarray(exc_logical, dtype=np.float64)
    inh_logical = np.asarray(inh_current)
    inh_drive: npt.NDArray[np.float64]
    if inh_logical.ndim == 0:
        inh_drive = np.full(exc_drive.size, float(inh_logical), dtype=np.float64)
    elif inh_logical.ndim == 1 and inh_logical.size == exc_drive.size:
        inh_drive = np.ascontiguousarray(inh_logical, dtype=np.float64)
    else:
        raise ValueError("inh_current must be a scalar or match exc_current length")
    if not np.isfinite(exc_drive).all() or not np.isfinite(inh_drive).all():
        raise ValueError("current values must contain only finite values")
    if _lib is None:
        raise ImportError(
            f"libalpha.so not built. Run: cd {_LIB_PATH.parent} && "
            f"mojo build --emit shared-lib -o {_LIB_PATH.name} "
            "alpha.mojo --target-cpu x86-64-v3"
        )
    size = exc_drive.size
    v_trace: npt.NDArray[np.float64] = np.empty(size, dtype=np.float64)
    a_exc_trace: npt.NDArray[np.float64] = np.empty(size, dtype=np.float64)
    i_exc_trace: npt.NDArray[np.float64] = np.empty(size, dtype=np.float64)
    a_inh_trace: npt.NDArray[np.float64] = np.empty(size, dtype=np.float64)
    i_inh_trace: npt.NDArray[np.float64] = np.empty(size, dtype=np.float64)
    spikes: npt.NDArray[np.float64] = np.empty(size, dtype=np.float64)
    finals: dict[str, npt.NDArray[np.float64]] = {
        name: np.empty(1, dtype=np.float64)
        for name in ("v", "a_exc", "i_exc", "a_inh", "i_inh", "spike_count")
    }
    status = _lib.alpha_simulate_c(
        ctypes.c_int32(size),
        *(
            float(value)
            for value in (
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
            )
        ),
        exc_drive.ctypes.data,
        inh_drive.ctypes.data,
        v_trace.ctypes.data,
        a_exc_trace.ctypes.data,
        i_exc_trace.ctypes.data,
        a_inh_trace.ctypes.data,
        i_inh_trace.ctypes.data,
        spikes.ctypes.data,
        finals["v"].ctypes.data,
        finals["a_exc"].ctypes.data,
        finals["i_exc"].ctypes.data,
        finals["a_inh"].ctypes.data,
        finals["i_inh"].ctypes.data,
        finals["spike_count"].ctypes.data,
    )
    if status == 4:
        raise FloatingPointError("alpha_simulate_c rejected exact-flow candidate")
    if status == 2:
        raise ValueError("alpha_simulate_c rejected configuration")
    if status == 3:
        raise ValueError("alpha_simulate_c rejected current")
    if status != 0:
        raise RuntimeError(f"alpha_simulate_c rejected code {status}")
    return {
        "v": v_trace,
        "a_exc": a_exc_trace,
        "i_exc": i_exc_trace,
        "a_inh": a_inh_trace,
        "i_inh": i_inh_trace,
        "spikes": spikes,
        "v_final": float(finals["v"][0]),
        "a_exc_final": float(finals["a_exc"][0]),
        "i_exc_final": float(finals["i_exc"][0]),
        "a_inh_final": float(finals["a_inh"][0]),
        "i_inh_final": float(finals["i_inh"][0]),
        "spike_count": int(finals["spike_count"][0]),
    }


__all__ = ["_HAS_MOJO_ALPHA", "simulate_alpha"]
