# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Array guards for native FFI boundaries

"""Validate NumPy array layout before zero-copy native bridge calls.

The native bridge, debugging tools, and stochastic-doctor diagnostics use this
module to reject non-contiguous or misaligned arrays before passing buffers into
Rust, C, Mojo, or other FFI-style paths.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def require_c_contiguous(
    arr: Any,
    name: str,
    dtype: np.dtype[Any] | type[np.generic] | None = None,
) -> np.ndarray[Any, Any]:
    """Validate array layout before a zero-copy native call."""
    expected_dtype = np.dtype(dtype) if dtype is not None else None

    if isinstance(arr, np.ndarray):
        if not arr.flags["C_CONTIGUOUS"]:
            raise ValueError(f"{name} must be C-contiguous; pass np.ascontiguousarray(x)")
        if not arr.flags["ALIGNED"]:
            raise ValueError(f"{name} is not aligned")
        if expected_dtype is not None and arr.dtype != expected_dtype:
            return arr.astype(expected_dtype, copy=False)
        return arr

    converted = np.asarray(arr, dtype=expected_dtype)
    if not converted.flags["C_CONTIGUOUS"]:
        raise ValueError(f"{name} must be C-contiguous; pass np.ascontiguousarray(x)")
    if not converted.flags["ALIGNED"]:
        raise ValueError(f"{name} is not aligned")
    return converted
