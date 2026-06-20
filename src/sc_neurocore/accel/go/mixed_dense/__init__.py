# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Go-backed mixed-precision dense MAC (ctypes dispatch)

"""Python entry point for the Go-compiled mixed-precision dense kernel.

Build::

    cd src/sc_neurocore/accel/go/mixed_dense
    PATH=/usr/local/go/bin:$PATH go build -buildmode=c-shared \\
        -o libmixed_dense.so mixed_dense.go

The ``.so`` is platform-specific and gitignored; the ``.go`` source and the
generated ``.h`` header are tracked. ``_HAS_GO_MIXED_DENSE`` is ``True`` iff the
shared library is present at import time.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

_LIB_PATH = Path(__file__).resolve().parent / "libmixed_dense.so"
_lib: ctypes.CDLL | None

try:
    _lib = ctypes.CDLL(str(_LIB_PATH))
    _lib.mixed_dense_forward_batch_q88_q1616_c.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,  # n_outputs, n_inputs, n_batch
        ctypes.c_void_p,
        ctypes.c_void_p,  # weights, inputs
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,  # outputs, overflow, underflow
    ]
    _lib.mixed_dense_forward_batch_q88_q1616_c.restype = ctypes.c_int
    _HAS_GO_MIXED_DENSE = True
except OSError:
    _lib = None
    _HAS_GO_MIXED_DENSE = False


def mixed_dense_forward_batch(
    weights_q88: npt.ArrayLike,
    inputs_q1616: npt.ArrayLike,
    n_outputs: int,
    n_inputs: int,
) -> dict[str, npt.NDArray[Any]]:
    """Go-accelerated batched mixed-precision dense MAC.

    Parameters
    ----------
    weights_q88, inputs_q1616, n_outputs, n_inputs
        See
        :func:`sc_neurocore.compiler.mixed_dense_kernel.mixed_dense_forward_batch_q88_q1616`.

    Returns
    -------
    dict
        Flat ``outputs_q1616``, ``overflow`` and ``underflow`` arrays of length
        ``n_batch * n_outputs`` — bit-identical to the Python floor.

    Raises
    ------
    ImportError
        If ``libmixed_dense.so`` is not built.
    ValueError
        If shapes are inconsistent.
    """
    if _lib is None:
        raise ImportError(
            f"libmixed_dense.so not built. Run: cd {_LIB_PATH.parent} && "
            f"go build -buildmode=c-shared -o {_LIB_PATH.name} mixed_dense.go"
        )
    weights = np.ascontiguousarray(weights_q88, dtype=np.int16).reshape(-1)
    inputs = np.ascontiguousarray(inputs_q1616, dtype=np.int32).reshape(-1)
    if weights.size != int(n_outputs) * int(n_inputs):
        raise ValueError(f"weights length must be n_outputs * n_inputs: got {weights.size}")
    if inputs.size == 0 or inputs.size % int(n_inputs) != 0:
        raise ValueError(f"inputs length {inputs.size} is not a multiple of n_inputs {n_inputs}")
    n_batch = inputs.size // int(n_inputs)
    count = n_batch * int(n_outputs)

    outputs: npt.NDArray[np.int32] = np.empty(count, dtype=np.int32)
    overflow: npt.NDArray[np.uint8] = np.empty(count, dtype=np.uint8)
    underflow: npt.NDArray[np.uint8] = np.empty(count, dtype=np.uint8)

    rc = _lib.mixed_dense_forward_batch_q88_q1616_c(
        ctypes.c_int(int(n_outputs)),
        ctypes.c_int(int(n_inputs)),
        ctypes.c_int(n_batch),
        weights.ctypes.data,
        inputs.ctypes.data,
        outputs.ctypes.data,
        overflow.ctypes.data,
        underflow.ctypes.data,
    )
    if rc != 0:
        raise ValueError(f"mixed_dense_forward_batch_q88_q1616_c returned non-zero: {rc}")

    return {
        "outputs_q1616": outputs,
        "overflow": overflow.astype(np.bool_),
        "underflow": underflow.astype(np.bool_),
    }


__all__ = ["mixed_dense_forward_batch"]
