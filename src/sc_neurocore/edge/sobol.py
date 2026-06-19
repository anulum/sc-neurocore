# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Sobol low-discrepancy sequence (ported from tinysc_riscv/sobol.rs)

"""Sobol low-discrepancy sequence generator for SC bitstream decorrelation.

Provides better uniformity than LFSR-16 at the cost of slightly more
compute per step. Uses Gray-code acceleration for O(1) per-sample
generation (no matrix multiply needed).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class SobolGenerator:
    """1D Sobol sequence generator with 16-bit resolution.

    Uses Joe-Kuo direction numbers (dimension 1) and Gray-code indexing
    so only one XOR per step.
    """

    DIRECTION_NUMBERS = np.array(
        [
            0x8000,
            0x4000,
            0x2000,
            0x1000,
            0x0800,
            0x0400,
            0x0200,
            0x0100,
            0x0080,
            0x0040,
            0x0020,
            0x0010,
            0x0008,
            0x0004,
            0x0002,
            0x0001,
        ],
        dtype=np.uint16,
    )

    def __init__(self, seed: int = 0):
        self._reg = np.uint16(seed)
        self._index = np.uint32(0)

    def step(self) -> int:
        """Advance by one step, return the next Sobol value in [0, 65535]."""
        c = 0
        idx = int(self._index)
        if idx > 0:
            c = (idx & -idx).bit_length() - 1
        if c < 16:
            self._reg ^= self.DIRECTION_NUMBERS[c]
        self._index += np.uint32(1)
        return int(self._reg)

    def encode(self, threshold: int, length: int) -> np.ndarray[Any, Any]:
        """Encode a probability into packed u64 words using Sobol sequence.

        Parameters
        ----------
        threshold : int
            Value in [0, 65535]. Each Sobol sample < threshold becomes a 1-bit.
        length : int
            Number of bits in the bitstream.

        Returns
        -------
        np.ndarray[Any, Any]
            Packed u64 bitstream array.
        """
        n_words = (length + 63) // 64
        out = np.zeros(n_words, dtype=np.uint64)
        for i in range(length):
            val = self.step()
            if val < threshold:
                out[i // 64] |= np.uint64(1) << np.uint64(i % 64)
        return out

    def reset(self, seed: int = 0) -> None:
        """Reset to initial state."""
        self._reg = np.uint16(seed)
        self._index = np.uint32(0)
