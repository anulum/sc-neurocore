# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Unified quasi-random RTL emitter (Sobol + Halton)

"""Unified emitter for quasi-random stochastic number generators.

Consolidates Sobol-16 and Halton-16 into a single factory interface,
allowing the compiler to select the optimal SNG backend based on
area/quality tradeoffs:

- **Sobol**: Superior discrepancy (O(1/N)), needs direction-number casez
- **Halton**: Nearly as good (O(log N / N)), zero LUT cost (pure bit-reversal)

Usage::

    from sc_neurocore.hdl_gen.quasirandom_emitter import QuasiRandomEmitter

    emitter = QuasiRandomEmitter(method="halton")
    verilog = emitter.generate()
"""

from __future__ import annotations

import logging
from typing import Literal

from ._ident import sanitize_ident
from .sobol16_emitter import Sobol16Emitter

logger = logging.getLogger(__name__)


class Halton16Emitter:
    """Emit a synthesisable standalone Halton-16 (Van der Corput base-2) module.

    Architecture: pure counter + bit-reversal wiring.
    Zero multipliers, zero LUTs for core logic.
    """

    def __init__(
        self,
        module_name: str = "sc_halton16_source",
    ) -> None:
        self.module_name = sanitize_ident(module_name, context="module name")

    def generate(self) -> str:
        """Return the standalone Halton-16 Verilog module."""
        lines = [
            f"module {self.module_name} (",
            "    input wire clk,",
            "    input wire rst_n,",
            "    input wire enable,",
            "    output reg [15:0] quasi_random,",
            "    output reg valid",
            ");",
            "",
            "    reg [15:0] counter;",
            "",
            "    // Bit-reversal = Van der Corput base-2 radical inverse",
            "    // Pure routing — zero LUT cost",
            "    wire [15:0] reversed;",
            "",
        ]

        # Generate bit-reversal wiring
        for i in range(16):
            lines.append(f"    assign reversed[{i}] = counter[{15 - i}];")

        lines.extend([
            "",
            "    always @(posedge clk or negedge rst_n) begin",
            "        if (!rst_n) begin",
            "            counter      <= 16'd0;",
            "            quasi_random <= 16'd0;",
            "            valid        <= 1'b0;",
            "        end else if (enable) begin",
            "            quasi_random <= reversed;",
            "            valid        <= 1'b1;",
            "            counter      <= counter + 16'd1;",
            "        end else begin",
            "            valid <= 1'b0;",
            "        end",
            "    end",
            "endmodule",
        ])
        return "\n".join(lines)


class QuasiRandomEmitter:
    """Unified factory for quasi-random SNG emitters.

    Parameters
    ----------
    method : str
        ``"sobol"`` or ``"halton"``.
    module_name : str, optional
        Override the default module name.
    seed : int, optional
        Seed for Sobol (ignored for Halton).
    """

    METHODS = {"sobol", "halton"}

    def __init__(
        self,
        method: Literal["sobol", "halton"] = "sobol",
        module_name: str | None = None,
        seed: int = 0,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(
                f"Unknown quasi-random method {method!r}. "
                f"Supported: {sorted(self.METHODS)}"
            )
        self.method = method
        self._seed = seed

        if method == "sobol":
            name = module_name or "sc_sobol16_source"
            self._emitter = Sobol16Emitter(module_name=name, seed=seed)
        else:
            name = module_name or "sc_halton16_source"
            self._emitter = Halton16Emitter(module_name=name)

        logger.debug(
            "QuasiRandomEmitter: method=%s, module=%s", method, name
        )

    def generate(self) -> str:
        """Generate the Verilog source for the selected method."""
        return self._emitter.generate()

    @property
    def module_name(self) -> str:
        """Return the sanitised module name."""
        return self._emitter.module_name
