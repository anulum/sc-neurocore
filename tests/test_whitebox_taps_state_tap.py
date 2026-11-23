# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStateTap from former test_whitebox_taps.py

"""Focused suite: TestStateTap from former test_whitebox_taps.py."""

from __future__ import annotations

from sc_neurocore.compiler.whitebox_taps import (
    StateTap,
)


class TestStateTap:
    """The white-box state tap value type."""

    def test_declaration_sized_signed(self) -> None:
        """A sized signed tap declares a signed bus."""
        tap = StateTap("v_state", "v_reg", msb="DATA_WIDTH-1", signed=True)
        assert tap.declaration() == "output wire signed [DATA_WIDTH-1:0] v_state"

    def test_declaration_scalar_unsigned(self) -> None:
        """A scalar tap declares a single wire."""
        assert StateTap("flag", "ready").declaration() == "output wire flag"

    def test_assignment(self) -> None:
        """A tap assigns its source expression."""
        assert StateTap("t", "32'd0", msb="31").assignment() == "    assign t = 32'd0;"
