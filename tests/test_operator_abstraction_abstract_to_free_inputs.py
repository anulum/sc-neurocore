# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAbstractToFreeInputs from former test_operator_abstraction.py

"""Focused suite: TestAbstractToFreeInputs from former test_operator_abstraction.py."""

from __future__ import annotations

from tests.operator_abstraction_support import *  # noqa: F403

class TestAbstractToFreeInputs:
    """The lifting transform."""

    def test_lifts_and_renames(self) -> None:
        out = abstract_to_free_inputs(
            _MODULE,
            top="foo",
            signals=[LiftedSignal("prod", "prod_in", msb="2*W-1", signed=True)],
        )
        # The product is now a free input, and its multiply driver is gone.
        ports = parse_module_interface(out, "foo", params={"W": 8})
        names = {p.name: p.direction for p in ports}
        assert names["prod_in"] == "input"
        assert next(p.width for p in ports if p.name == "prod_in") == 16
        assert "a * b" not in out
        assert "assign prod" not in out
        # Downstream uses were rewired to the new port name.
        assert "assign scaled = prod_in >>> 2;" in out

    def test_lifts_inline_wire_initializer(self) -> None:
        # ``wire NAME = EXPR;`` — the declaration itself is the driver (QIF style).
        out = abstract_to_free_inputs(
            _MODULE_INLINE,
            top="baz",
            signals=[LiftedSignal("sq", "sq_in", msb="2*W-1", signed=True)],
        )
        ports = parse_module_interface(out, "baz", params={"W": 8})
        assert any(p.name == "sq_in" and p.direction == "input" for p in ports)
        assert "a * a" not in out
        # The inline declaration is gone; the downstream use is rewired to the port.
        assert "sq_in = " not in out
        assert "scaled = sq_in >>> 2" in out

    def test_lifts_without_rename(self) -> None:
        out = abstract_to_free_inputs(
            _MODULE, top="foo", signals=[LiftedSignal("prod", "prod", msb="2*W-1", signed=True)]
        )
        assert any(
            p.name == "prod" and p.direction == "input"
            for p in parse_module_interface(out, "foo", params={"W": 8})
        )
        assert "a * b" not in out

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            abstract_to_free_inputs(_MODULE, top="foo", signals=[])

    def test_rejects_duplicate_ports(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            abstract_to_free_inputs(
                _MODULE,
                top="foo",
                signals=[
                    LiftedSignal("prod", "p", msb="2*W-1"),
                    LiftedSignal("scaled", "p", msb="W-1"),
                ],
            )

    def test_rejects_collision_with_existing_port(self) -> None:
        with pytest.raises(ValueError, match="already exists"):
            abstract_to_free_inputs(
                _MODULE, top="foo", signals=[LiftedSignal("prod", "a", msb="2*W-1")]
            )

    def test_missing_declaration_raises(self) -> None:
        # ``zz`` is an implicit internal net — a continuous assign but no wire decl.
        src = (
            "module bar(input wire clk, output reg y);\n"
            "    assign zz = clk;\n"
            "    always @(posedge clk) y <= zz;\n"
            "endmodule\n"
        )
        with pytest.raises(ValueError, match="declaration"):
            abstract_to_free_inputs(src, top="bar", signals=[LiftedSignal("zz", "zz_in")])

    def test_missing_driver_raises(self) -> None:
        # ``w`` is declared but never driven by a continuous assign.
        src = "module bar(input wire clk); wire [7:0] w; endmodule"
        with pytest.raises(ValueError, match="driver"):
            abstract_to_free_inputs(src, top="bar", signals=[LiftedSignal("w", "w_in", msb="7")])
