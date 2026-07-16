# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo-backed MPR mean-field batch

"""Typed ctypes facade for the Mojo MPR shared library."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import numpy.typing as npt

from sc_neurocore.neurons.models.ermentrout_kopell_pop import (
    ErmentroutKopellPopulationResult,
)

_ACCEL_ROOT = Path(__file__).resolve().parents[2]
_LIB_PATH = _ACCEL_ROOT / "mojo" / "ermentrout_kopell_pop" / "libermentrout_kopell_pop.so"
_MAX_NATIVE_STEPS = (1 << 31) - 1


def _configure_library(library: ctypes.CDLL) -> ctypes.CDLL:
    library.ermentrout_kopell_pop_simulate_c.argtypes = [
        ctypes.c_int32,
        *([ctypes.c_double] * 7),
        *([ctypes.c_void_p] * 5),
    ]
    library.ermentrout_kopell_pop_simulate_c.restype = ctypes.c_int32
    return library


def _load_library() -> tuple[ctypes.CDLL | None, bool]:
    try:
        return _configure_library(ctypes.CDLL(str(_LIB_PATH))), True
    except OSError:
        return None, False


_lib, _HAS_MOJO_ERMENTROUT_KOPELL_POP = _load_library()


def simulate_ermentrout_kopell_pop(
    r_init: float,
    v_init: float,
    tau: float,
    delta: float,
    eta_bar: float,
    coupling: float,
    dt: float,
    ext_input: npt.ArrayLike,
) -> ErmentroutKopellPopulationResult:
    """Run the Mojo implementation of the complete MPR recurrence.

    Parameters
    ----------
    r_init, v_init : float
        Initial population firing rate and mean membrane potential.
    tau, delta, eta_bar, coupling, dt : float
        Complete MPR configuration and explicit-Euler step.
    ext_input : ArrayLike
        One finite external drive value per step.

    Returns
    -------
    dict[str, numpy.ndarray | float]
        Post-update ``r`` and ``v`` traces plus both final-state receipts.

    Raises
    ------
    ValueError
        If the drive is not a finite one-dimensional vector.
    ImportError
        If the maintained Mojo shared library is unavailable.
    RuntimeError
        If the C ABI rejects the configuration or input.
    FloatingPointError
        If a finite valid-entry recurrence produces an invalid candidate state.

    Notes
    -----
    The Mojo kernel performs a validation pass before its output pass, so any
    rejected batch leaves all caller buffers unchanged.
    """
    logical = np.asarray(ext_input)
    if logical.ndim != 1:
        raise ValueError(f"ext_input must be one-dimensional: got shape {logical.shape}")
    if logical.size > _MAX_NATIVE_STEPS:
        raise ValueError(f"ext_input exceeds the signed-32-bit step limit: {logical.size}")
    drive = np.ascontiguousarray(logical, dtype=np.float64)
    if not np.isfinite(drive).all():
        raise ValueError("ext_input must contain only finite values")
    if _lib is None:
        raise ImportError(
            "libermentrout_kopell_pop.so not built. Run: mojo build --emit shared-lib "
            f"-o {_LIB_PATH} {_LIB_PATH.parent / 'ermentrout_kopell_pop.mojo'} "
            "--target-cpu x86-64-v3"
        )
    r_trace = np.empty(drive.size, dtype=np.float64)
    v_trace = np.empty(drive.size, dtype=np.float64)
    r_final: npt.NDArray[np.float64] = np.empty(1, dtype=np.float64)
    v_final: npt.NDArray[np.float64] = np.empty(1, dtype=np.float64)
    status = _lib.ermentrout_kopell_pop_simulate_c(
        ctypes.c_int32(drive.size),
        *(float(value) for value in (r_init, v_init, tau, delta, eta_bar, coupling, dt)),
        drive.ctypes.data,
        r_trace.ctypes.data,
        v_trace.ctypes.data,
        r_final.ctypes.data,
        v_final.ctypes.data,
    )
    if status == 4:
        raise FloatingPointError("ermentrout_kopell_pop_simulate_c rejected candidate state")
    if status != 0:
        raise RuntimeError(f"ermentrout_kopell_pop_simulate_c rejected code {status}")
    return {
        "r": r_trace,
        "v": v_trace,
        "r_final": float(r_final[0]),
        "v_final": float(v_final[0]),
    }


__all__ = ["_HAS_MOJO_ERMENTROUT_KOPELL_POP", "simulate_ermentrout_kopell_pop"]
