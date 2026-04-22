# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Research Kuramoto HDL emitter for bounded synthesis experiments

from __future__ import annotations

import math

from ._ident import sanitize_ident


class KuramotoEmitter:
    """Emit a bounded fixed-point Kuramoto phase core for HDL experiments.

    The generated RTL is intentionally narrow in scope:

    - noiseless only
    - all-to-all scalar coupling only
    - fixed-point phase state
    - LUT-based sine approximation

    This is a synthesis exploration scaffold, not a drop-in replacement for
    the production Kuramoto solvers.
    """

    def __init__(
        self,
        module_name: str = "sc_kuramoto_phase_core",
        *,
        n_oscillators: int = 4,
        omegas: list[float] | tuple[float, ...] | None = None,
        initial_phases: list[float] | tuple[float, ...] | None = None,
        coupling: float = 0.1,
        dt: float = 1e-2,
        data_width: int = 24,
        fraction: int = 16,
        lut_size: int = 64,
    ) -> None:
        if n_oscillators < 1:
            raise ValueError("n_oscillators must be >= 1")
        if data_width < 16:
            raise ValueError("data_width must be >= 16")
        if not 0 < fraction < data_width:
            raise ValueError("fraction must satisfy 0 < fraction < data_width")
        if lut_size < 16 or lut_size & (lut_size - 1):
            raise ValueError("lut_size must be a power of two >= 16")

        self.module_name = sanitize_ident(module_name, context="module name")
        self.n_oscillators = n_oscillators
        self.data_width = data_width
        self.fraction = fraction
        self.lut_size = lut_size
        self.dt = dt
        self.coupling = coupling
        self.omegas = list(omegas) if omegas is not None else [1.0] * n_oscillators
        self.initial_phases = (
            list(initial_phases) if initial_phases is not None else [0.0] * n_oscillators
        )

        if len(self.omegas) != n_oscillators:
            raise ValueError("omegas length must equal n_oscillators")
        if len(self.initial_phases) != n_oscillators:
            raise ValueError("initial_phases length must equal n_oscillators")

    def _fixed_int(self, value: float) -> int:
        return int(round(value * (1 << self.fraction)))

    def _signed_literal(self, value: int) -> str:
        magnitude = abs(value)
        if value < 0:
            return f"-{self.data_width}'sd{magnitude}"
        return f"{self.data_width}'sd{magnitude}"

    def _lut_lines(self) -> list[str]:
        lines = [
            "    function automatic signed [DATA_WIDTH-1:0] sin_lut;",
            "        input signed [DATA_WIDTH-1:0] phase_value;",
            "        reg signed [DATA_WIDTH-1:0] wrapped_phase;",
            "        reg [7:0] lut_index;",
            "        begin",
            "            wrapped_phase = wrap_phase(phase_value);",
            "            lut_index = (wrapped_phase * LUT_SIZE) / PHASE_MODULUS;",
            "            case (lut_index)",
        ]
        for idx in range(self.lut_size):
            value = self._fixed_int(math.sin((2.0 * math.pi * idx) / self.lut_size))
            lines.append(f"                8'd{idx}: sin_lut = {self._signed_literal(value)};")
        lines.extend(
            [
                "                default: sin_lut = 0;",
                "            endcase",
                "        end",
                "    endfunction",
            ]
        )
        return lines

    def generate(self) -> str:
        phase_modulus = self._fixed_int(2.0 * math.pi)
        half_phase_modulus = self._fixed_int(math.pi)
        dt_fixed = self._fixed_int(self.dt)
        coupling_fixed = self._fixed_int(self.coupling / self.n_oscillators)
        acc_width = self.data_width + max(4, math.ceil(math.log2(max(2, self.n_oscillators))) + 2)

        lines = [
            f"module {self.module_name} (",
            "    input wire clk,",
            "    input wire rst_n,",
            "    input wire step_en,",
            "    output reg update_done,",
            f"    output wire [{self.n_oscillators * self.data_width - 1}:0] phase_bus",
            ");",
            "",
            "    // Research boundary: fixed-point noiseless Kuramoto phase core.",
            "    // This module keeps only the all-to-all scalar-coupling regime and",
            "    // does not attempt to cover the production Rust solver extensions.",
            f"    localparam integer N_OSC = {self.n_oscillators};",
            f"    localparam integer DATA_WIDTH = {self.data_width};",
            f"    localparam integer FRACTION = {self.fraction};",
            f"    localparam integer LUT_SIZE = {self.lut_size};",
            f"    localparam integer ACC_WIDTH = {acc_width};",
            f"    localparam signed [DATA_WIDTH-1:0] PHASE_MODULUS = {self._signed_literal(phase_modulus)};",
            f"    localparam signed [DATA_WIDTH-1:0] HALF_PHASE_MODULUS = {self._signed_literal(half_phase_modulus)};",
            f"    localparam signed [DATA_WIDTH-1:0] DT = {self._signed_literal(dt_fixed)};",
            f"    localparam signed [DATA_WIDTH-1:0] K_OVER_N = {self._signed_literal(coupling_fixed)};",
            "",
            "    function automatic signed [DATA_WIDTH-1:0] wrap_phase;",
            "        input signed [DATA_WIDTH-1:0] phase_value;",
            "        reg signed [DATA_WIDTH-1:0] wrapped;",
            "        begin",
            "            wrapped = phase_value;",
            "            if (wrapped >= PHASE_MODULUS) begin",
            "                wrapped = wrapped - PHASE_MODULUS;",
            "            end else if (wrapped < 0) begin",
            "                wrapped = wrapped + PHASE_MODULUS;",
            "            end",
            "            wrap_phase = wrapped;",
            "        end",
            "    endfunction",
            "",
            "    function automatic signed [DATA_WIDTH-1:0] wrap_delta;",
            "        input signed [DATA_WIDTH-1:0] delta_value;",
            "        reg signed [DATA_WIDTH-1:0] wrapped;",
            "        begin",
            "            wrapped = delta_value;",
            "            if (wrapped > HALF_PHASE_MODULUS) begin",
            "                wrapped = wrapped - PHASE_MODULUS;",
            "            end else if (wrapped < -HALF_PHASE_MODULUS) begin",
            "                wrapped = wrapped + PHASE_MODULUS;",
            "            end",
            "            wrap_delta = wrapped;",
            "        end",
            "    endfunction",
            "",
        ]
        lines.extend(self._lut_lines())
        lines.append("")

        for idx, omega in enumerate(self.omegas):
            lines.append(
                f"    localparam signed [DATA_WIDTH-1:0] OMEGA_{idx} = "
                f"{self._signed_literal(self._fixed_int(omega))};"
            )
        for idx, phase in enumerate(self.initial_phases):
            lines.append(
                f"    localparam signed [DATA_WIDTH-1:0] INIT_PHASE_{idx} = "
                f"{self._signed_literal(self._fixed_int(phase % (2.0 * math.pi)))};"
            )
        lines.append("")

        for idx in range(self.n_oscillators):
            lines.append(f"    reg signed [DATA_WIDTH-1:0] phase_reg_{idx};")
        lines.append("")

        for row in range(self.n_oscillators):
            terms: list[str] = []
            for col in range(self.n_oscillators):
                lines.append(
                    f"    wire signed [DATA_WIDTH-1:0] phase_diff_{row}_{col} = "
                    f"wrap_delta(phase_reg_{col} - phase_reg_{row});"
                )
                lines.append(
                    f"    wire signed [DATA_WIDTH-1:0] sin_term_{row}_{col} = "
                    f"sin_lut(phase_diff_{row}_{col});"
                )
                if col != row:
                    terms.append(f"sin_term_{row}_{col}")
            sum_expr = " + ".join(terms) if terms else f"{acc_width}'sd0"
            lines.extend(
                [
                    f"    wire signed [ACC_WIDTH-1:0] coupling_sum_{row} = {sum_expr};",
                    f"    wire signed [DATA_WIDTH+ACC_WIDTH-1:0] coupling_mult_{row} = coupling_sum_{row} * K_OVER_N;",
                    f"    wire signed [DATA_WIDTH-1:0] coupling_term_{row} = coupling_mult_{row} >>> FRACTION;",
                    f"    wire signed [DATA_WIDTH-1:0] phase_velocity_{row} = OMEGA_{row} + coupling_term_{row};",
                    f"    wire signed [2*DATA_WIDTH-1:0] delta_mult_{row} = phase_velocity_{row} * DT;",
                    f"    wire signed [DATA_WIDTH-1:0] phase_delta_{row} = delta_mult_{row} >>> FRACTION;",
                    f"    wire signed [DATA_WIDTH-1:0] next_phase_{row} = "
                    f"wrap_phase(phase_reg_{row} + phase_delta_{row});",
                    "",
                ]
            )

        for idx in range(self.n_oscillators):
            lines.append(
                f"    assign phase_bus[{(idx + 1) * self.data_width - 1}:{idx * self.data_width}] = phase_reg_{idx};"
            )
        lines.extend(
            [
                "",
                "    always @(posedge clk or negedge rst_n) begin",
                "        if (!rst_n) begin",
            ]
        )
        for idx in range(self.n_oscillators):
            lines.append(f"            phase_reg_{idx} <= INIT_PHASE_{idx};")
        lines.extend(
            [
                "            update_done <= 1'b0;",
                "        end else begin",
                "            update_done <= 1'b0;",
                "            if (step_en) begin",
            ]
        )
        for idx in range(self.n_oscillators):
            lines.append(f"                phase_reg_{idx} <= next_phase_{idx};")
        lines.extend(
            [
                "                update_done <= 1'b1;",
                "            end",
                "        end",
                "    end",
                "endmodule",
            ]
        )
        return "\n".join(lines) + "\n"
