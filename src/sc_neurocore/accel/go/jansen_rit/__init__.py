# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Go-backed Jansen–Rit equation-(6) batch

"""Typed ctypes facade for the Go Jansen–Rit shared library."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import numpy.typing as npt

from sc_neurocore.neurons.models.jansen_rit import JansenRitResult

_LIB_PATH = Path(__file__).resolve().parent / "libjansen_rit.so"


def _configure_library(library: ctypes.CDLL) -> ctypes.CDLL:
    """Attach the complete equation-(6) C signature to a loaded library."""
    library.jansen_rit_simulate_c.argtypes = [
        ctypes.c_int,
        *([ctypes.c_double] * 15),
        *([ctypes.c_void_p] * 8),
        *([ctypes.POINTER(ctypes.c_double)] * 6),
    ]
    library.jansen_rit_simulate_c.restype = ctypes.c_int
    return library


def _load_library() -> tuple[ctypes.CDLL | None, bool]:
    try:
        return _configure_library(ctypes.CDLL(str(_LIB_PATH))), True
    except OSError:
        return None, False


_lib, _HAS_GO_JANSEN_RIT = _load_library()


def simulate_jansen_rit(
    y0_init: float,
    y3_init: float,
    y1_init: float,
    y4_init: float,
    y2_init: float,
    y5_init: float,
    a_exc: float,
    b_exc: float,
    a_rate: float,
    b_rate: float,
    c: float,
    e0: float,
    v0: float,
    r: float,
    dt: float,
    p_ext: npt.ArrayLike,
) -> JansenRitResult:
    """Run the Go implementation of the equation-(6) recurrence."""
    drive = np.ascontiguousarray(p_ext, dtype=np.float64)
    if drive.ndim != 1:
        raise ValueError(f"p_ext must be one-dimensional: got shape {drive.shape}")
    if not np.isfinite(drive).all():
        raise ValueError("p_ext must contain only finite values")
    if _lib is None:
        raise ImportError(
            f"libjansen_rit.so not built. Run: cd {_LIB_PATH.parent} && "
            f"go build -buildmode=c-shared -o {_LIB_PATH.name} jansen_rit.go"
        )
    traces = [np.empty(drive.size, dtype=np.float64) for _ in range(7)]
    finals = [ctypes.c_double(0.0) for _ in range(6)]
    status = _lib.jansen_rit_simulate_c(
        ctypes.c_int(drive.size),
        *(
            ctypes.c_double(value)
            for value in (
                y0_init,
                y3_init,
                y1_init,
                y4_init,
                y2_init,
                y5_init,
                a_exc,
                b_exc,
                a_rate,
                b_rate,
                c,
                e0,
                v0,
                r,
                dt,
            )
        ),
        drive.ctypes.data,
        *(trace.ctypes.data for trace in traces),
        *(ctypes.byref(final) for final in finals),
    )
    if status != 0:
        raise RuntimeError(f"jansen_rit_simulate_c rejected the contract with code {status}")
    keys = ("y0", "y3", "y1", "y4", "y2", "y5", "eeg")
    result: JansenRitResult = {key: trace for key, trace in zip(keys, traces, strict=True)}
    for key, final in zip(keys[:6], finals, strict=True):
        result[f"{key}_final"] = final.value
    return result


__all__ = ["_HAS_GO_JANSEN_RIT", "simulate_jansen_rit"]
