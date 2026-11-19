# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCC from former test_tinysc_ports.py

"""Focused suite: TestSCC from former test_tinysc_ports.py."""

from __future__ import annotations

import pytest

from sc_neurocore.edge.bitstream import scc


class TestSCC:
    """Check packed stochastic correlation over logical bits."""

    def test_identical(self) -> None:
        """Identical alternating streams are fully correlated."""
        a = [0xAAAA_AAAA]
        assert abs(scc(a, a, 32) - 1.0) < 0.01

    def test_anticorrelated(self) -> None:
        """Complementary alternating streams are anticorrelated."""
        a = [0xAAAA_AAAA]
        b = [0x5555_5555]
        assert abs(scc(a, b, 32) - (-1.0)) < 0.01

    def test_zero_length(self) -> None:
        """The empty logical prefix has zero correlation."""
        assert scc([0], [0], 0) == 0.0

    def test_zero_density_streams_hit_numerator_floor(self) -> None:
        """Zero-density streams have zero correlation."""
        # Empty (all-zero) streams over a non-zero length give pa=pb=p_and=0,
        # so the numerator collapses to the |num|<eps floor: the coefficient
        # is defined as 0 rather than 0/0.
        assert scc([0x0000_0000], [0x0000_0000], 32) == 0.0

    def test_deterministic_single_bit_ignores_padding(self) -> None:
        """The identical logical bit has zero covariance and finite SCC."""
        assert scc([0b11], [0b01], 1) == 0.0

    def test_mismatched_inputs_are_refused(self) -> None:
        """Correlation requires matching storage for both streams."""
        with pytest.raises(ValueError, match="lengths"):
            scc([0x5], [0xA, 0], 4)

    def test_correlation_ignores_tail_padding(self) -> None:
        """Opposite bit patterns stay anticorrelated at a partial word boundary."""
        assert scc([0x5], [0xFFFF_FFFA], 4) == -1.0
