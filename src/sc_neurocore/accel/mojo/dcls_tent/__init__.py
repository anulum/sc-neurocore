# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo-backed DCLS-max Q8.8 tent kernel (ctypes dispatch)

"""Python entry point for the Mojo-compiled DCLS-max tent kernel.

Build::

    cd src/sc_neurocore/accel/mojo/dcls_tent
    mojo build --emit shared-lib -o libdcls_tent.so dcls_tent.mojo

The ``.so`` is platform-specific and gitignored; the ``.mojo`` source is tracked.
``_HAS_MOJO_DCLS`` is ``True`` iff the shared library is present at import time.

The kernel is exact integer Q8.8 arithmetic with no transcendental path, so —
unlike the floating-point Mojo neuron kernels that tolerate last-ULP libm drift —
this backend is bit-identical to the Rust, Julia, Go and Python references.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

_LIB_PATH = Path(__file__).resolve().parent / "libdcls_tent.so"
_lib: ctypes.CDLL | None

try:
    _lib = ctypes.CDLL(str(_LIB_PATH))
    _lib.dcls_max_forward_batch_q88_c.argtypes = [
        ctypes.c_long,
        ctypes.c_long,  # n_channels, n_taps
        ctypes.c_void_p,
        ctypes.c_void_p,  # spikes, weights
        ctypes.c_void_p,
        ctypes.c_void_p,  # centres, sigmas
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,  # outputs, accumulators, overflow
        ctypes.c_void_p,
        ctypes.c_void_p,  # active_counts, max_gates
    ]
    _lib.dcls_max_forward_batch_q88_c.restype = ctypes.c_long
    _HAS_MOJO_DCLS = True
except OSError:
    _lib = None
    _HAS_MOJO_DCLS = False


def dcls_max_forward_batch(
    spikes: npt.ArrayLike,
    weights_q88: npt.ArrayLike,
    centres_q88: npt.ArrayLike,
    sigmas_q88: npt.ArrayLike,
    n_taps: int,
) -> dict[str, npt.NDArray[Any]]:
    """Mojo-accelerated batched DCLS-max tent kernel.

    Parameters
    ----------
    spikes, weights_q88, centres_q88, sigmas_q88, n_taps
        See :func:`sc_neurocore.scpn.dcls_tent_kernel.dcls_max_forward_batch_q88`.

    Returns
    -------
    dict
        Mapping with ``outputs_q88``, ``accumulators_q16_16``, ``overflow``,
        ``active_tap_counts`` and ``max_gates_q88`` arrays — bit-identical to the
        Python floor.

    Raises
    ------
    ImportError
        If ``libdcls_tent.so`` is not built.
    ValueError
        If shapes are inconsistent or any sigma is non-positive.
    """
    if _lib is None:
        raise ImportError(
            f"libdcls_tent.so not built. Run: cd {_LIB_PATH.parent} && "
            f"mojo build --emit shared-lib -o {_LIB_PATH.name} dcls_tent.mojo"
        )
    spike_arr = np.ascontiguousarray(spikes, dtype=np.uint8).reshape(-1)
    weight_arr = np.ascontiguousarray(weights_q88, dtype=np.int16).reshape(-1)
    centre_arr = np.ascontiguousarray(centres_q88, dtype=np.int16).reshape(-1)
    sigma_arr = np.ascontiguousarray(sigmas_q88, dtype=np.int16).reshape(-1)
    n_channels = int(centre_arr.size)
    if sigma_arr.size != n_channels:
        raise ValueError(f"centres/sigmas length mismatch: {n_channels} vs {sigma_arr.size}")
    expected = n_channels * int(n_taps)
    if spike_arr.size != expected or weight_arr.size != expected:
        raise ValueError(
            "DCLS batch spike/weight length must be n_channels * n_taps "
            f"({expected}): spikes={spike_arr.size}, weights={weight_arr.size}"
        )

    outputs: npt.NDArray[np.int16] = np.empty(n_channels, dtype=np.int16)
    accumulators: npt.NDArray[np.int32] = np.empty(n_channels, dtype=np.int32)
    overflow: npt.NDArray[np.uint8] = np.empty(n_channels, dtype=np.uint8)
    active_counts: npt.NDArray[np.int64] = np.empty(n_channels, dtype=np.int64)
    max_gates: npt.NDArray[np.int16] = np.empty(n_channels, dtype=np.int16)

    rc = _lib.dcls_max_forward_batch_q88_c(
        ctypes.c_long(n_channels),
        ctypes.c_long(int(n_taps)),
        spike_arr.ctypes.data,
        weight_arr.ctypes.data,
        centre_arr.ctypes.data,
        sigma_arr.ctypes.data,
        outputs.ctypes.data,
        accumulators.ctypes.data,
        overflow.ctypes.data,
        active_counts.ctypes.data,
        max_gates.ctypes.data,
    )
    if rc != 0:
        raise ValueError(f"Mojo dcls_max_forward_batch_q88_c returned non-zero: {rc}")

    return {
        "outputs_q88": outputs,
        "accumulators_q16_16": accumulators,
        "overflow": overflow.astype(np.bool_),
        "active_tap_counts": active_counts,
        "max_gates_q88": max_gates,
    }


__all__ = ["dcls_max_forward_batch"]
