# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia-backed mixed-precision dense MAC (juliacall dispatch)

"""Python entry point for the Julia mixed-precision Q8.8×Q16.16 dense kernel.

The integer dense MAC is exact, so the Julia backend is bit-identical to the
Rust, Go, Mojo and Python references. Julia boots lazily via ``juliacall``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

try:
    from juliacall import Main as _jl

    _HAS_JULIA_MIXED_DENSE = True
except ImportError:
    _jl = None
    _HAS_JULIA_MIXED_DENSE = False


_KERNEL_DIR = Path(__file__).resolve().parent
_LOADED = False


def _ensure_loaded() -> Any:
    """Include ``mixed_dense.jl`` into Julia ``Main`` on first use; return the module."""
    global _LOADED
    if _jl is None:
        raise ImportError("juliacall not available; install the `julia` extra")
    if not _LOADED:
        jl_path = _KERNEL_DIR / "mixed_dense.jl"
        if not jl_path.is_file():
            raise FileNotFoundError(f"mixed_dense.jl missing at {jl_path}")
        _jl.include(str(jl_path))
        _LOADED = True
    return _jl.MixedDenseAccel


def mixed_dense_forward_batch(
    weights_q88: npt.ArrayLike,
    inputs_q1616: npt.ArrayLike,
    n_outputs: int,
    n_inputs: int,
) -> dict[str, npt.NDArray[Any]]:
    """Julia-accelerated batched mixed-precision dense MAC.

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
    """
    mod = _ensure_loaded()
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

    mod.mixed_dense_forward_batch_q88_q1616_b(
        weights, inputs, int(n_outputs), int(n_inputs), outputs, overflow, underflow
    )
    return {
        "outputs_q1616": outputs,
        "overflow": overflow.astype(np.bool_),
        "underflow": underflow.astype(np.bool_),
    }


__all__ = ["mixed_dense_forward_batch"]
