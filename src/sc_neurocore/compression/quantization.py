# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Weight and delay quantization for compression

"""Quantize weights and delays for reduced hardware precision.

Weight quantization: reduce from float64 to fixed-point with configurable
bit width. Fewer bits = smaller BRAM and simpler multiplier circuits.

Delay quantization: round continuous delays to integer steps or coarser
grids. Fewer delay levels = smaller delay buffers on FPGA.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def quantize_weights(
    weights: list[np.ndarray[Any, Any]],
    bits: int = 8,
    symmetric: bool = True,
) -> list[np.ndarray[Any, Any]]:
    """Quantize weight matrices to fixed-point with given bit width.

    Parameters
    ----------
    weights : list of ndarray
        Float weight matrices.
    bits : int
        Target bit width (default 8). Range: [2, 16].
    symmetric : bool
        Symmetric quantization around zero (default True).

    Returns
    -------
    list of ndarray
        Quantized weights (still float dtype but with discrete values).
    """
    bits = max(2, min(bits, 16))
    n_levels = 2**bits

    quantized = []
    for w in weights:
        if symmetric:
            abs_max = max(np.abs(w).max(), 1e-8)
            scale = abs_max / (n_levels // 2 - 1)
            q = np.round(w / scale) * scale
            q = np.clip(q, -abs_max, abs_max)
        else:
            w_min, w_max = w.min(), w.max()
            w_range = max(w_max - w_min, 1e-8)
            scale = w_range / (n_levels - 1)
            q = np.round((w - w_min) / scale) * scale + w_min

        quantized.append(q)

    return quantized


def quantize_delays(
    delays: np.ndarray[Any, Any],
    resolution: int = 1,
    max_delay: int | None = None,
) -> np.ndarray[Any, Any]:
    """Quantize continuous delays to integer grid.

    Parameters
    ----------
    delays : ndarray
        Continuous delay values.
    resolution : int
        Delay step size (default 1). Resolution=2 means delays are
        rounded to {0, 2, 4, 6, ...}, halving the buffer depth.
    max_delay : int, optional
        Clamp delays to this maximum.

    Returns
    -------
    ndarray of int
        Quantized integer delays.
    """
    q = np.round(delays / resolution).astype(np.int64) * resolution
    q = np.clip(q, 0, None)
    if max_delay is not None:
        q = np.clip(q, 0, max_delay)
    return q
