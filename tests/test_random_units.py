# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Philox4x32-10 reference against an independent implementation

"""Tests of :mod:`sc_neurocore.compiler.random_units`.

``randomgen``'s compiled Philox (``number=4, width=32``) is the independent
implementation. Its ``random_raw`` yields the four 32-bit words of each block in
order and increments the counter before the first block, so the block for
counter ``c`` is its first output when started at ``c - 1``; starting at
``2**128 - 1`` wraps to the block of counter 0. Skipped when ``randomgen`` is not
installed.
"""

from __future__ import annotations

import pytest

from sc_neurocore.compiler.random_units import counter_words, philox4x32

_CASES = [
    (0, 0),
    (0, 1),
    (0x123456789ABCDEF0, 7),
    ((1 << 64) - 1, (1 << 128) - 2),
    (0xA4093822299F31D0, 0x03707344_13198A2E_85A308D3_243F6A88),
]


@pytest.mark.parametrize(("key", "counter"), _CASES)
def test_blocks_equal_randomgen(key: int, counter: int) -> None:
    """Four consecutive blocks equal randomgen's output for the same key and counters."""
    randomgen = pytest.importorskip("randomgen")
    generator = randomgen.Philox(number=4, width=32, key=key, counter=(counter - 1) % (1 << 128))
    expected = [int(word) for word in generator.random_raw(16)]
    key_words = (key & 0xFFFFFFFF, key >> 32)
    mine: list[int] = []
    for step in range(4):
        mine.extend(philox4x32(counter_words((counter + step) % (1 << 128)), key_words))
    assert mine == expected


def test_word_and_round_ranges_are_checked() -> None:
    """Words outside 32 bits and round counts outside [1, 16] are refused."""
    with pytest.raises(ValueError, match="32-bit"):
        philox4x32((1 << 32, 0, 0, 0), (0, 0))
    with pytest.raises(ValueError, match="1 to 16 rounds"):
        philox4x32((0, 0, 0, 0), (0, 0), rounds=0)
    with pytest.raises(ValueError, match="128 bits"):
        counter_words(1 << 128)


def test_counter_words_split_little_endian_without_an_optional_reference() -> None:
    """The public split and refusal contract runs even when randomgen is absent."""
    assert counter_words((4 << 96) | (3 << 64) | (2 << 32) | 1) == (1, 2, 3, 4)
    assert counter_words((1 << 128) - 1) == ((1 << 32) - 1,) * 4
    with pytest.raises(ValueError, match="128 bits"):
        counter_words(-1)
