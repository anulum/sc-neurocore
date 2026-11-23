# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — typed ctypes facade for the Mojo contrastive/ssl library

"""Typed ctypes facade for the Mojo contrastive self-supervised learning kernels.

The library exports the InfoNCE loss, one CSDP phase and the goodness score of
:mod:`sc_neurocore.contrastive.ssl`. The dispatcher
:mod:`sc_neurocore.accel.contrastive_ssl` validates every input before it calls
these functions, so a non-zero status here is an internal contract breach and
raises :class:`RuntimeError`.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.mojo.native_library import load_mojo_library

_ACCEL_ROOT = Path(__file__).resolve().parents[2]
_LIB_PATH = _ACCEL_ROOT / "mojo" / "contrastive_ssl" / "libcontrastive_ssl.so"
_BUILD_HINT = (
    f"cd {_LIB_PATH.parent} && mojo build --emit shared-lib -o libcontrastive_ssl.so "
    "contrastive_ssl.mojo --target-cpu x86-64-v3"
)
_MAX_COUNT = (1 << 31) - 1


def _configure_library(library: ctypes.CDLL) -> ctypes.CDLL:
    library.contrastive_ssl_infonce_c.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    library.contrastive_ssl_infonce_c.restype = ctypes.c_int32
    library.contrastive_ssl_csdp_c.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int32,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    library.contrastive_ssl_csdp_c.restype = ctypes.c_int32
    library.contrastive_ssl_goodness_c.argtypes = [ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p]
    library.contrastive_ssl_goodness_c.restype = ctypes.c_int32
    return library


def _load_library() -> tuple[ctypes.CDLL | None, bool]:
    try:
        return _configure_library(load_mojo_library(str(_LIB_PATH))), True
    except OSError:
        return None, False


_lib, _HAS_MOJO_CONTRASTIVE_SSL = _load_library()


def _library() -> ctypes.CDLL:
    if _lib is None:
        raise ImportError(f"libcontrastive_ssl.so not built. Run: {_BUILD_HINT}")
    return _lib


def _address(array: npt.NDArray[np.float64]) -> int:
    return int(array.ctypes.data)


def _check(status: int, kernel: str) -> None:
    if status != 0:
        raise RuntimeError(f"Mojo {kernel} refused validated input with status {status}")


def infonce_loss(
    view_a: npt.NDArray[np.float64],
    view_b: npt.NDArray[np.float64],
    temperature: float,
) -> float:
    """Return the mean InfoNCE loss of two validated (batch, features) views.

    Parameters
    ----------
    view_a, view_b : ndarray of shape (batch, features), float64
        C-contiguous, finite views of the same shape, ``features >= 1``.
    temperature : float
        Finite, positive temperature.

    Returns
    -------
    float
        The loss; ``0.0`` for a batch of fewer than two rows.

    Raises
    ------
    ImportError
        If the shared library is not built.
    ValueError
        If the batch or feature count exceeds the signed 32-bit limit.
    RuntimeError
        If the kernel refuses the validated input.
    """
    batch, features = view_a.shape
    if batch > _MAX_COUNT or features > _MAX_COUNT:
        raise ValueError("view shape exceeds the signed-32-bit limit of the Mojo lane")
    loss = np.zeros(1, dtype=np.float64)
    status = _library().contrastive_ssl_infonce_c(
        batch, features, temperature, _address(view_a), _address(view_b), _address(loss)
    )
    _check(int(status), "InfoNCE")
    return float(loss[0])


def csdp_update(
    weights: npt.NDArray[np.float64],
    pre: npt.NDArray[np.float64],
    post: npt.NDArray[np.float64],
    lr: float,
    decay: float,
    *,
    positive: bool,
) -> npt.NDArray[np.float64]:
    """Return one CSDP phase applied to validated weights.

    Parameters
    ----------
    weights : ndarray of shape (n_post, n_pre), float64
        C-contiguous, finite weights.
    pre, post : ndarray of shape (n_pre,) and (n_post,), float64
        C-contiguous, finite spike-rate vectors.
    lr, decay : float
        Finite, non-negative rule parameters.
    positive : bool
        ``True`` for the Hebbian phase, ``False`` for the anti-Hebbian phase.

    Returns
    -------
    ndarray of shape (n_post, n_pre)
        The updated weights, in a new array.

    Raises
    ------
    ImportError
        If the shared library is not built.
    ValueError
        If a dimension exceeds the signed 32-bit limit.
    RuntimeError
        If the kernel refuses the validated input.
    """
    n_post, n_pre = weights.shape
    if n_post > _MAX_COUNT or n_pre > _MAX_COUNT:
        raise ValueError("weight shape exceeds the signed-32-bit limit of the Mojo lane")
    out: npt.NDArray[np.float64] = np.empty_like(weights)
    status = _library().contrastive_ssl_csdp_c(
        n_post,
        n_pre,
        lr,
        decay,
        1 if positive else 0,
        _address(weights),
        _address(pre),
        _address(post),
        _address(out),
    )
    _check(int(status), "CSDP")
    return out


def goodness(values: npt.NDArray[np.float64]) -> float:
    """Return the sum of squared validated activations.

    Parameters
    ----------
    values : ndarray, float64
        C-contiguous, finite activations of any shape.

    Returns
    -------
    float
        ``sum(values**2)``.

    Raises
    ------
    ImportError
        If the shared library is not built.
    RuntimeError
        If the kernel refuses the validated input.
    """
    result = np.zeros(1, dtype=np.float64)
    status = _library().contrastive_ssl_goodness_c(values.size, _address(values), _address(result))
    _check(int(status), "goodness")
    return float(result[0])
