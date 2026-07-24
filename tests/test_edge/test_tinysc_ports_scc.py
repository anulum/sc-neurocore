# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCC from former test_tinysc_ports.py

"""Focused suite: TestSCC from former test_tinysc_ports.py."""

from __future__ import annotations

from tinysc_ports_support import *  # noqa: F403


class TestSCC:
    def test_identical(self):
        a = [0xAAAA_AAAA]
        assert abs(scc(a, a, 32) - 1.0) < 0.01

    def test_anticorrelated(self):
        a = [0xAAAA_AAAA]
        b = [0x5555_5555]
        assert abs(scc(a, b, 32) - (-1.0)) < 0.01

    def test_zero_length(self):
        assert scc([0], [0], 0) == 0.0

    def test_zero_density_streams_hit_numerator_floor(self):
        # Empty (all-zero) streams over a non-zero length give pa=pb=p_and=0,
        # so the numerator collapses to the |num|<eps floor: the coefficient
        # is defined as 0 rather than 0/0.
        assert scc([0x0000_0000], [0x0000_0000], 32) == 0.0

    def test_under_counted_length_hits_denominator_floor(self):
        # bit_length under-counts the bits packed into the words, pushing pa
        # above 1 and breaking the p_and<=min(pa,pb) invariant: here pa=2, pb=1
        # make the denominator collapse to 0 while the numerator stays nonzero,
        # exercising the |denom|<eps floor that keeps the result finite.
        assert scc([0b11], [0b01], 1) == 0.0
