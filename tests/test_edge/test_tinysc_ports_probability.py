# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestProbability from former test_tinysc_ports.py

"""Focused suite: TestProbability from former test_tinysc_ports.py."""

from __future__ import annotations

from typing import cast

import pytest

from sc_neurocore.edge.bitstream import MASK32, probability


class TestProbability:
    """Check probabilities and logical lengths of packed streams."""

    def test_all_ones(self) -> None:
        """Count every bit of a full word of ones."""
        assert abs(probability([MASK32], 32) - 1.0) < 1e-6

    def test_all_zeros(self) -> None:
        """A full word of zeros has zero probability."""
        assert abs(probability([0], 32) - 0.0) < 1e-6

    def test_half(self) -> None:
        """An alternating full word has probability one half."""
        assert abs(probability([0xAAAA_AAAA], 32) - 0.5) < 1e-6

    def test_zero_length(self) -> None:
        """An empty logical prefix has zero probability."""
        assert probability([MASK32], 0) == 0.0

    def test_logical_tail_does_not_count_padding(self) -> None:
        """Count only the logical prefix of the packed input."""
        assert probability([MASK32], 8) == 1.0
        assert probability([0, MASK32, MASK32], 33) == 1.0 / 33.0

    def test_length_beyond_buffer_is_refused(self) -> None:
        """A short buffer cannot supply the advertised bit length."""
        with pytest.raises(ValueError, match="bit_length"):
            probability([MASK32], 33)

    @pytest.mark.parametrize("bit_length", [-1, 33, cast(int, True), cast(int, 2.5)])
    def test_invalid_length_is_refused(self, bit_length: int) -> None:
        """Packed bit lengths must be integer counts within the buffer."""
        with pytest.raises(ValueError, match="bit_length"):
            probability([MASK32], bit_length)
