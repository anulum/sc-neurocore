# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMask from former test_behavioral_equivalence.py

"""Focused suite: TestMask from former test_behavioral_equivalence.py."""

from __future__ import annotations

from sc_neurocore.neurons.fixed_point_lif import (
    _mask,
)


class TestMask:
    """Mask integers to a signed bit width the way the Verilog datapath does."""

    def test_positive(self) -> None:
        """A positive value inside the range is kept."""
        assert _mask(100, 16) == 100

    def test_negative(self) -> None:
        """A negative value inside the range is kept."""
        assert _mask(-100, 16) == -100

    def test_overflow_wraps(self) -> None:
        """One past the 16-bit maximum wraps to the minimum."""
        # 32768 in 16-bit signed is -32768
        assert _mask(32768, 16) == -32768

    def test_underflow_wraps(self) -> None:
        """One below the 16-bit minimum wraps to the maximum."""
        # -32769 in 16-bit signed wraps to 32767
        assert _mask(-32769, 16) == 32767
