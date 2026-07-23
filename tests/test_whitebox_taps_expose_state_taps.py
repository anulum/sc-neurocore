# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExposeStateTaps from former test_whitebox_taps.py

"""Focused suite: TestExposeStateTaps from former test_whitebox_taps.py."""

from __future__ import annotations

from tests.whitebox_taps_support import *  # noqa: F403

class TestExposeStateTaps:
    """The instrumentation transform."""

    def test_adds_ports_and_assigns(self) -> None:
        out = expose_state_taps(
            _MODULE,
            top="foo",
            taps=[
                StateTap("s_tap", "s", msb="W-1", signed=False),
                StateTap("flag", "1'b0"),
            ],
        )
        # Original ports survive, taps are added as outputs.
        ports = parse_module_interface(out, "foo", params={"W": 8})
        names = {p.name: p.direction for p in ports}
        assert names["clk"] == "input"
        assert names["y"] == "output"
        assert names["s_tap"] == "output"
        assert names["flag"] == "output"
        assert next(p.width for p in ports if p.name == "s_tap") == 8
        # Continuous assigns drive the taps before endmodule.
        assert "assign s_tap = s;" in out
        assert "assign flag = 1'b0;" in out
        assert out.rindex("assign flag = 1'b0;") < out.rindex("endmodule")
        # The original datapath is untouched.
        assert "always @(posedge clk) begin s <= x; y <= s; end" in out

    def test_rejects_empty_taps(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            expose_state_taps(_MODULE, top="foo", taps=[])

    def test_rejects_duplicate_tap_names(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            expose_state_taps(
                _MODULE,
                top="foo",
                taps=[StateTap("t", "s", msb="W-1"), StateTap("t", "x", msb="W-1")],
            )

    def test_rejects_collision_with_existing_port(self) -> None:
        with pytest.raises(ValueError, match="already exists"):
            expose_state_taps(_MODULE, top="foo", taps=[StateTap("y", "s", msb="W-1")])

    def test_missing_endmodule_raises(self) -> None:
        truncated = "module bar(input wire clk); reg r; assign x = r;"
        with pytest.raises(ValueError, match="endmodule"):
            expose_state_taps(truncated, top="bar", taps=[StateTap("t", "r")])

    def test_unknown_module_raises(self) -> None:
        with pytest.raises(ValueError, match="not found"):
            expose_state_taps(_MODULE, top="nope", taps=[StateTap("t", "s", msb="W-1")])
