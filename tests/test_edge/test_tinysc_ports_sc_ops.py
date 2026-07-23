# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCOps from former test_tinysc_ports.py

"""Focused suite: TestSCOps from former test_tinysc_ports.py."""

from __future__ import annotations

from tinysc_ports_support import *  # noqa: F403

class TestSCOps:
    def test_sc_and(self):
        assert sc_and(0b1010, 0b1100) == 0b1000

    def test_sc_or(self):
        assert sc_or(0b1010, 0b0101) == 0b1111

    def test_sc_xor(self):
        assert sc_xor(0b1010, 0b1100) == 0b0110

    def test_sc_sub(self):
        assert sc_sub(0b1110, 0b0110) == 0b1000

    def test_sc_mux(self):
        assert sc_mux(0xFF, 0x00, 0x0F) == 0x0F

    def test_and_packed(self):
        a = [0xAAAA_AAAA, 0xFFFF_FFFF]
        b = [0x5555_5555, 0x0000_FFFF]
        out = and_packed(a, b)
        assert out[0] == 0
        assert out[1] == 0x0000_FFFF

    def test_mux_packed(self):
        a = [0xFFFF_FFFF]
        b = [0x0000_0000]
        s = [0x0000_FFFF]
        out = mux_packed(a, b, s)
        assert out[0] == 0x0000_FFFF

    def test_and_packed_rejects_length_mismatch(self):
        with pytest.raises(AssertionError):
            and_packed([0x1, 0x2], [0x1])

    def test_mux_packed_rejects_length_mismatch(self):
        with pytest.raises(AssertionError):
            mux_packed([0x1], [0x0, 0x1], [0x1])
