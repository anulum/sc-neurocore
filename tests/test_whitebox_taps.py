# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for whitebox state-tap instrumentation

"""Tests for exposing internal signals as observation output ports.

These are pure text-transformation tests — no formal toolchain required. The
end-to-end k-induction proof that consumes the taps lives with the equivalence
runner tests and self-skips without ``sby``.
"""

from __future__ import annotations

import pytest

from sc_neurocore.compiler.equivalence_miter import parse_module_interface
from sc_neurocore.compiler.whitebox_taps import StateTap, expose_state_taps

_MODULE = """`timescale 1ns/1ps
module foo #(parameter integer W = 8)(
    input wire clk,
    input wire [W-1:0] x,
    output reg [W-1:0] y
);
    reg [W-1:0] s;
    always @(posedge clk) begin s <= x; y <= s; end
endmodule
"""


class TestStateTap:
    """The tap value type."""

    def test_declaration_sized_signed(self) -> None:
        tap = StateTap("v_state", "v_reg", msb="DATA_WIDTH-1", signed=True)
        assert tap.declaration() == "output wire signed [DATA_WIDTH-1:0] v_state"

    def test_declaration_scalar_unsigned(self) -> None:
        assert StateTap("flag", "ready").declaration() == "output wire flag"

    def test_assignment(self) -> None:
        assert StateTap("t", "32'd0", msb="31").assignment() == "    assign t = 32'd0;"


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
