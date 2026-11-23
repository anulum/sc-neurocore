# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestParseModuleInterface from former test_equivalence_miter.py

"""Focused suite: TestParseModuleInterface from former test_equivalence_miter.py."""

from __future__ import annotations

import pytest
from sc_neurocore.compiler.equivalence_miter import (
    MiterPort,
    parse_module_interface,
)
from tests.equivalence_miter_support import (
    _LIF_REFERENCE,
)


class TestParseModuleInterface:
    """Parsing the ANSI port interface, resolving parameter-dependent widths."""

    def test_parses_directions_widths_and_signedness(self) -> None:
        """Each port is read with its direction, width and signedness."""
        ports = parse_module_interface(
            _LIF_REFERENCE, "sc_lif_reference", params={"DATA_WIDTH": 16}
        )
        by_name = {p.name: p for p in ports}
        assert by_name["clk"] == MiterPort("clk", 1, False, "input")
        assert by_name["rst_n"] == MiterPort("rst_n", 1, False, "input")
        assert by_name["leak_k"] == MiterPort("leak_k", 16, True, "input")
        assert by_name["I_t"] == MiterPort("I_t", 16, True, "input")
        assert by_name["spike_out"] == MiterPort("spike_out", 1, False, "output")
        assert by_name["v_out"] == MiterPort("v_out", 16, True, "output")

    def test_declaration_order_is_preserved(self) -> None:
        """Ports come back in declaration order."""
        ports = parse_module_interface(
            _LIF_REFERENCE, "sc_lif_reference", params={"DATA_WIDTH": 16}
        )
        assert [p.name for p in ports] == [
            "clk",
            "rst_n",
            "leak_k",
            "I_t",
            "spike_out",
            "v_out",
        ]

    def test_width_expression_uses_parameter_value(self) -> None:
        """A width expression uses the given parameter value."""
        ports = parse_module_interface(_LIF_REFERENCE, "sc_lif_reference", params={"DATA_WIDTH": 8})
        assert {p.name: p.width for p in ports}["v_out"] == 8

    def test_missing_module_raises(self) -> None:
        """Asking for a module that is not in the source raises."""
        with pytest.raises(ValueError, match="not found"):
            parse_module_interface(_LIF_REFERENCE, "no_such_module")

    def test_unknown_width_parameter_raises(self) -> None:
        """A width that names an unset parameter raises."""
        with pytest.raises(ValueError, match="unknown parameter"):
            parse_module_interface(_LIF_REFERENCE, "sc_lif_reference")

    def test_module_without_ports_raises(self) -> None:
        """A module with no ports raises."""
        with pytest.raises(ValueError, match="no ports|port list"):
            parse_module_interface("module empty(); endmodule", "empty")

    def test_shift_and_arithmetic_in_width(self) -> None:
        """Arithmetic and shifts in a port range evaluate to the right widths."""
        src = "module m(input wire [ (2*4) - 1 : 0 ] a, output wire [1<<3:0] b); endmodule"
        ports = parse_module_interface(src, "m")
        widths = {p.name: p.width for p in ports}
        assert widths["a"] == 8  # (2*4) - 1 downto 0 -> 8 bits
        assert widths["b"] == 9  # (1<<3)=8 downto 0 -> 9 bits
