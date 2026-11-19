# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCOps from former test_tinysc_ports.py

"""Focused suite: TestSCOps from former test_tinysc_ports.py."""

from __future__ import annotations

import pytest

from sc_neurocore.edge.bitstream import (
    and_packed,
    mux_packed,
    sc_and,
    sc_mux,
    sc_or,
    sc_sub,
    sc_xor,
)


class TestSCOps:
    """Check word operations and packed operand lengths."""

    def test_sc_and(self) -> None:
        """Retain only positions where both input words are set."""
        assert sc_and(0b1010, 0b1100) == 0b1000

    def test_sc_or(self) -> None:
        """Retain positions where either input word is set."""
        assert sc_or(0b1010, 0b0101) == 0b1111

    def test_sc_xor(self) -> None:
        """Retain positions where the input word bits differ."""
        assert sc_xor(0b1010, 0b1100) == 0b0110

    def test_sc_sub(self) -> None:
        """Clear positions in the first word selected by the second."""
        assert sc_sub(0b1110, 0b0110) == 0b1000

    def test_sc_mux(self) -> None:
        """Select corresponding bits across the input words."""
        assert sc_mux(0xFF, 0x00, 0x0F) == 0x0F

    def test_and_packed(self) -> None:
        """Apply bitwise conjunction across both packed words."""
        a = [0xAAAA_AAAA, 0xFFFF_FFFF]
        b = [0x5555_5555, 0x0000_FFFF]
        out = and_packed(a, b)
        assert out[0] == 0
        assert out[1] == 0x0000_FFFF

    def test_mux_packed(self) -> None:
        """Apply the select word to the packed operands."""
        a = [0xFFFF_FFFF]
        b = [0x0000_0000]
        s = [0x0000_FFFF]
        out = mux_packed(a, b, s)
        assert out[0] == 0x0000_FFFF

    def test_and_packed_rejects_length_mismatch(self) -> None:
        """Reject packed conjunction with mismatched lengths."""
        with pytest.raises(ValueError):
            and_packed([0x1, 0x2], [0x1])

    def test_mux_packed_rejects_length_mismatch(self) -> None:
        """Reject packed selection with mismatched lengths."""
        with pytest.raises(ValueError):
            mux_packed([0x1], [0x0, 0x1], [0x1])
