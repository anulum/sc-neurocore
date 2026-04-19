# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — LFSR-16 Encoder (ported from tinysc_riscv/lfsr.rs)

"""Deterministic LFSR-16 encoder bit-compatible with core_engine::Lfsr16.

Polynomial: x^16 + x^14 + x^13 + x^11 + 1 (maximal length = 65535).
Generates packed u32-word bitstreams from probability thresholds.
"""

from __future__ import annotations

from .bitstream import MASK32


class Lfsr16:
    """16-bit Galois LFSR bitstream encoder.

    Bit-compatible with the Rust core_engine::bitstream::Lfsr16.
    Uses u32-packed output for MCU word alignment.
    """

    TAPS = 0xD008  # x^16+x^14+x^13+x^11+1

    def __init__(self, seed: int = 0xACE1):
        self.reg = seed & 0xFFFF
        if self.reg == 0:
            self.reg = 0xACE1

    def step(self) -> int:
        """Advance LFSR by one clock, return new state."""
        bit = ((self.reg >> 0) ^ (self.reg >> 2) ^ (self.reg >> 3) ^ (self.reg >> 5)) & 1
        self.reg = ((self.reg >> 1) | (bit << 15)) & 0xFFFF
        return self.reg

    def encode(self, threshold: int, bit_length: int) -> list[int]:
        """Encode probability (threshold/65535) into packed u32 words.

        Parameters
        ----------
        threshold : int
            Comparison threshold [0, 65535]. Higher = more 1-bits.
        bit_length : int
            Number of bits in the output bitstream.

        Returns
        -------
        list[int]
            Packed u32 words representing the bitstream.
        """
        n_words = (bit_length + 31) // 32
        out = [0] * n_words
        for i in range(bit_length):
            val = self.step()
            if val < threshold:
                out[i // 32] |= 1 << (i % 32)
        return [w & MASK32 for w in out]

    def encode_float(self, p: float, bit_length: int) -> list[int]:
        """Encode a probability [0.0, 1.0] into a packed bitstream."""
        threshold = int(p * 65535)
        return self.encode(threshold, bit_length)
