# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Standalone Sobol-16 RTL emitter with compare-before-advance semantics

"""Standalone RTL emitter for the 16-bit Sobol stochastic source."""

from __future__ import annotations

from ._ident import sanitize_ident


class Sobol16Emitter:
    """Emit a synthesisable standalone Sobol-16 Verilog module."""

    def __init__(self, module_name: str = "sc_sobol16_source", seed: int = 0) -> None:
        self.module_name = sanitize_ident(module_name, context="module name")
        self.seed = seed & 0xFFFF

    def generate(self) -> str:
        """Return the standalone Sobol-16 Verilog module."""
        seed_hex = f"16'h{self.seed:04X}"
        lines = [
            f"module {self.module_name} (",
            "    input wire clk,",
            "    input wire rst_n,",
            "    input wire [15:0] threshold,",
            "    output wire bit_out,",
            "    output reg [15:0] value,",
            "    output reg [15:0] index",
            ");",
            "",
            "    reg [15:0] direction;",
            "",
            "    // Compare the current Sobol value before the next clocked advance.",
            "    assign bit_out = (value < threshold);",
            "",
            "    always @(*) begin",
            "        casez (index)",
            "            16'b???????????????1: direction = 16'h8000;",
            "            16'b??????????????10: direction = 16'h4000;",
            "            16'b?????????????100: direction = 16'h2000;",
            "            16'b????????????1000: direction = 16'h1000;",
            "            16'b???????????10000: direction = 16'h0800;",
            "            16'b??????????100000: direction = 16'h0400;",
            "            16'b?????????1000000: direction = 16'h0200;",
            "            16'b????????10000000: direction = 16'h0100;",
            "            16'b???????100000000: direction = 16'h0080;",
            "            16'b??????1000000000: direction = 16'h0040;",
            "            16'b?????10000000000: direction = 16'h0020;",
            "            16'b????100000000000: direction = 16'h0010;",
            "            16'b???1000000000000: direction = 16'h0008;",
            "            16'b??10000000000000: direction = 16'h0004;",
            "            16'b?100000000000000: direction = 16'h0002;",
            "            16'b1000000000000000: direction = 16'h0001;",
            "            default: direction = 16'h8000;",
            "        endcase",
            "    end",
            "",
            "    always @(posedge clk or negedge rst_n) begin",
            "        if (!rst_n) begin",
            f"            value <= {seed_hex};",
            "            index <= 16'd0;",
            "        end else begin",
            "            value <= value ^ direction;",
            "            index <= index + 16'd1;",
            "        end",
            "    end",
            "endmodule",
        ]
        return "\n".join(lines)
