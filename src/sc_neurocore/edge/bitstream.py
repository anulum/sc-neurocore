# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Packed Bitstream Primitives (ported from tinysc_riscv/bitstream.rs)

"""Packed u32-word bitstream operations for SC arithmetic.

All operations work on lists of unsigned 32-bit integers, mirroring
the bare-metal Rust implementation for RISC-V targets. Provides
popcount, SC AND/OR/XOR/MUX/SUB, SCC computation, and probability
estimation.
"""

from __future__ import annotations

MASK32 = 0xFFFF_FFFF


def popcount32(word: int) -> int:
    """Count set bits in a u32 word (Wilkes-Wheeler-Gill)."""
    x = word & MASK32
    x = x - ((x >> 1) & 0x5555_5555)
    x = (x & 0x3333_3333) + ((x >> 2) & 0x3333_3333)
    x = (x + (x >> 4)) & 0x0F0F_0F0F
    x = x + (x >> 8)
    x = x + (x >> 16)
    return x & 0x3F


def popcount_slice(words: list[int]) -> int:
    """Popcount over a packed u32 word slice."""
    total = 0
    for w in words:
        total += popcount32(w)
    return total


def sc_and(a: int, b: int) -> int:
    """SC multiply (bitwise AND)."""
    return (a & b) & MASK32


def sc_or(a: int, b: int) -> int:
    """SC saturating addition (bitwise OR)."""
    return (a | b) & MASK32


def sc_xor(a: int, b: int) -> int:
    """SC absolute difference / HDC bind (bitwise XOR)."""
    return (a ^ b) & MASK32


def sc_sub(a: int, b: int) -> int:
    """SC saturating subtraction: a AND NOT b."""
    return (a & (~b & MASK32)) & MASK32


def sc_mux(a: int, b: int, sel: int) -> int:
    """SC scaled addition (2:1 MUX): (a AND sel) OR (b AND NOT sel)."""
    return ((a & sel) | (b & (~sel & MASK32))) & MASK32


def and_packed(a: list[int], b: list[int]) -> list[int]:
    """SC AND over two packed word slices."""
    assert len(a) == len(b)
    return [(x & y) & MASK32 for x, y in zip(a, b)]


def mux_packed(a: list[int], b: list[int], sel: list[int]) -> list[int]:
    """SC MUX over two packed word slices with a select bitstream."""
    assert len(a) == len(b) == len(sel)
    return [((x & s) | (y & (~s & MASK32))) & MASK32 for x, y, s in zip(a, b, sel)]


def probability(words: list[int], bit_length: int) -> float:
    """Estimated probability from a packed bitstream."""
    if bit_length == 0:
        return 0.0
    return popcount_slice(words) / bit_length


def scc(a: list[int], b: list[int], bit_length: int) -> float:
    """SCC between two packed u32 bitstreams (Alaghi & Hayes, 2013).

    Returns a correlation coefficient in [-1, 1].
    """
    assert len(a) == len(b)
    if bit_length == 0:
        return 0.0
    n = float(bit_length)
    pa = popcount_slice(a) / n
    pb = popcount_slice(b) / n

    and_count = sum(popcount32(x & y) for x, y in zip(a, b))
    p_and = and_count / n

    num = p_and - (pa * pb)
    if abs(num) < 1e-7:
        return 0.0
    if num > 0.0:
        denom = min(pa, pb) - (pa * pb)
    else:
        denom = (pa * pb) - max(pa + pb - 1.0, 0.0)
    if abs(denom) < 1e-7:
        return 0.0
    return max(-1.0, min(1.0, num / denom))
