# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGapJunction from former test_gap_junction.py

"""Focused suite: TestGapJunction from former test_gap_junction.py."""

from __future__ import annotations

import pytest
from sc_neurocore.synapses.gap_junction import (
    GapJunction,
)


class TestGapJunction:
    """Electrical gap junctions."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("conductance", -0.1),
            ("conductance", float("nan")),
            ("rectification", -0.1),
            ("rectification", 1.1),
            ("rectification", float("inf")),
        ],
    )
    def test_invalid_parameters_fail_closed(self, field: str, value: float) -> None:
        """Each invalid junction parameter is rejected."""
        kwargs = {"conductance": 0.1, "rectification": 0.0}
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            GapJunction(**kwargs)

    def test_current_positive_dv(self) -> None:
        """A positive voltage difference drives a positive current."""
        gj = GapJunction(conductance=0.1)
        i = gj.current(v_pre=-50.0, v_post=-65.0)
        assert i == 0.1 * 15.0  # 1.5

    def test_current_negative_dv(self) -> None:
        """A negative voltage difference drives a negative current."""
        gj = GapJunction(conductance=0.1)
        i = gj.current(v_pre=-65.0, v_post=-50.0)
        assert i == 0.1 * -15.0  # -1.5

    def test_current_zero_dv(self) -> None:
        """Equal voltages carry no current."""
        gj = GapJunction(conductance=0.5)
        assert gj.current(v_pre=-60.0, v_post=-60.0) == 0.0

    def test_bidirectional_symmetry(self) -> None:
        """Reversing the direction reverses the current."""
        gj = GapJunction(conductance=0.2)
        i_forward = gj.current(v_pre=-50.0, v_post=-70.0)
        i_backward = gj.current(v_pre=-70.0, v_post=-50.0)
        assert i_forward == -i_backward

    def test_rectification(self) -> None:
        """Full rectification blocks the negative current."""
        gj = GapJunction(conductance=0.1, rectification=1.0)
        i_pos = gj.current(v_pre=-50.0, v_post=-65.0)  # dv > 0 → full current
        i_neg = gj.current(v_pre=-65.0, v_post=-50.0)  # dv < 0 → blocked
        assert i_pos == 0.1 * 15.0
        assert i_neg == 0.0

    def test_partial_rectification(self) -> None:
        """Partial rectification scales the negative current."""
        gj = GapJunction(conductance=1.0, rectification=0.5)
        i_neg = gj.current(v_pre=-70.0, v_post=-50.0)  # dv < 0 → reduced
        expected = 1.0 * (-20.0) * (1.0 - 0.5)  # = -10.0
        assert i_neg == expected

    @pytest.mark.parametrize(
        ("v_pre", "v_post"),
        [(float("nan"), -65.0), (-50.0, float("inf"))],
    )
    def test_invalid_current_voltage_fails_closed(self, v_pre: float, v_post: float) -> None:
        """A non-finite voltage is rejected."""
        gj = GapJunction(conductance=0.1)
        with pytest.raises(ValueError, match="voltage"):
            gj.current(v_pre=v_pre, v_post=v_post)
