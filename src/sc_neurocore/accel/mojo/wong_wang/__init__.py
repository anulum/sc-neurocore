# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo-backed Wong-Wang Euler/OU batch

"""Typed ctypes facade for the Mojo Wong-Wang shared library."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import numpy.typing as npt

WongWangMapping = dict[str, npt.NDArray[np.float64] | float]
_LIB_PATH = Path(__file__).resolve().parent / "libwong_wang.so"


def _configure_library(library: ctypes.CDLL) -> ctypes.CDLL:
    """Attach the complete Euler/OU C signature to a loaded library."""
    library.wong_wang_simulate_c.argtypes = [
        ctypes.c_int,
        *([ctypes.c_double] * 12),
        *([ctypes.c_void_p] * 13),
    ]
    library.wong_wang_simulate_c.restype = ctypes.c_int
    return library


def _load_library() -> tuple[ctypes.CDLL | None, bool]:
    """Load the platform-local Mojo shared object when present."""
    try:
        return _configure_library(ctypes.CDLL(str(_LIB_PATH))), True
    except OSError:
        return None, False


_lib, _HAS_MOJO_WONG_WANG = _load_library()


def _inputs(
    stim1: npt.ArrayLike,
    stim2: npt.ArrayLike,
    xi: npt.ArrayLike,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Convert and validate three one-dimensional input streams."""
    stim1_values = np.ascontiguousarray(stim1, dtype=np.float64)
    stim2_values = np.ascontiguousarray(stim2, dtype=np.float64)
    xi_values = np.ascontiguousarray(xi, dtype=np.float64)
    for name, array in (("stim1", stim1_values), ("stim2", stim2_values), ("xi", xi_values)):
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional: got shape {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
    steps = stim1_values.size
    if stim2_values.size != steps:
        raise ValueError(f"stim1 and stim2 length mismatch: {steps} vs {stim2_values.size}")
    if xi_values.size != 2 * steps:
        raise ValueError(f"xi length must be 2 * n_steps ({2 * steps}): got {xi_values.size}")
    return stim1_values, stim2_values, xi_values


def simulate_wong_wang(
    s1_init: float,
    s2_init: float,
    noise1_init: float,
    noise2_init: float,
    tau_s: float,
    tau_ampa: float,
    gamma: float,
    j_n: float,
    j_cross: float,
    i_0: float,
    sigma: float,
    dt: float,
    stim1: npt.ArrayLike,
    stim2: npt.ArrayLike,
    xi: npt.ArrayLike,
) -> WongWangMapping:
    """Run the Mojo implementation of the published Euler/OU recurrence.

    Parameters
    ----------
    s1_init, s2_init : float
        Initial NMDA gating fractions.
    noise1_init, noise2_init : float
        Initial AMPA Ornstein-Uhlenbeck current states.
    tau_s, tau_ampa, gamma, j_n, j_cross, i_0, sigma, dt : float
        Published reduced-model parameters.
    stim1, stim2 : ArrayLike
        Per-step external currents.
    xi : ArrayLike
        Interleaved standard-normal samples of length ``2 * n_steps``.

    Returns
    -------
    dict[str, numpy.ndarray | float]
        Six post-update traces and four final dynamic states.

    Raises
    ------
    ImportError
        If the Mojo shared library has not been built.
    ValueError
        If an input stream violates the public shape or finite-value contract.
    RuntimeError
        If the native kernel rejects a scalar or candidate state.
    """
    stim1_values, stim2_values, xi_values = _inputs(stim1, stim2, xi)
    if _lib is None:
        raise ImportError(
            f"libwong_wang.so not built. Run: cd {_LIB_PATH.parent} && "
            f"~/.pixi/bin/mojo build --emit shared-lib -o {_LIB_PATH.name} wong_wang.mojo"
        )
    steps = stim1_values.size
    traces = [np.empty(steps, dtype=np.float64) for _ in range(6)]
    finals: list[npt.NDArray[np.float64]] = [np.empty(1, dtype=np.float64) for _ in range(4)]
    rc = _lib.wong_wang_simulate_c(
        ctypes.c_int(steps),
        *(
            ctypes.c_double(value)
            for value in (
                s1_init,
                s2_init,
                noise1_init,
                noise2_init,
                tau_s,
                tau_ampa,
                gamma,
                j_n,
                j_cross,
                i_0,
                sigma,
                dt,
            )
        ),
        stim1_values.ctypes.data,
        stim2_values.ctypes.data,
        xi_values.ctypes.data,
        *(trace.ctypes.data for trace in traces),
        *(final.ctypes.data for final in finals),
    )
    if rc != 0:
        raise RuntimeError(f"Mojo wong_wang_simulate_c rejected the contract with code {rc}")
    return {
        "s1": traces[0],
        "s2": traces[1],
        "noise1": traces[2],
        "noise2": traces[3],
        "r1": traces[4],
        "r2": traces[5],
        "s1_final": float(finals[0][0]),
        "s2_final": float(finals[1][0]),
        "noise1_final": float(finals[2][0]),
        "noise2_final": float(finals[3][0]),
    }
