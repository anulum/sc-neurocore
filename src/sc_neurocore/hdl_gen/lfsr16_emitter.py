# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Standalone LFSR-16 RTL emitter with compare-before-advance semantics

"""Standalone RTL emitter for the canonical 16-bit stochastic source.

The emitted module exposes the current 16-bit state and a compare output
`bit_out = (state < threshold)`. This preserves the same compare-before-
advance semantics used by the software and Rust encoders.
"""

from __future__ import annotations

from ._ident import sanitize_ident


class Lfsr16Emitter:
    """Emit a synthesisable standalone LFSR-16 Verilog module."""

    def __init__(self, module_name: str = "sc_lfsr16_source", seed: int = 0xACE1) -> None:
        self.module_name = sanitize_ident(module_name, context="module name")
        self.seed = seed & 0xFFFF
        if self.seed == 0:
            self.seed = 0xACE1

    def generate(self) -> str:
        """Return the standalone LFSR-16 Verilog module."""
        seed_hex = f"16'h{self.seed:04X}"
        lines = [
            f"module {self.module_name} (",
            "    input wire clk,",
            "    input wire rst_n,",
            "    input wire [15:0] threshold,",
            "    output wire bit_out,",
            "    output reg [15:0] state",
            ");",
            "",
            f"    localparam [15:0] SEED = {seed_hex};",
            "    wire feedback;",
            "",
            "    // Compare the current state before the next clocked advance.",
            "    assign bit_out = (state < threshold);",
            "    assign feedback = state[0] ^ state[2] ^ state[3] ^ state[5];",
            "",
            "    always @(posedge clk or negedge rst_n) begin",
            "        if (!rst_n) begin",
            "            state <= SEED;",
            "        end else begin",
            "            state <= {feedback, state[15:1]};",
            "        end",
            "    end",
            "endmodule",
        ]
        return "\n".join(lines)
