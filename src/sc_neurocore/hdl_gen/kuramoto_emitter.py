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
        """Initialize a bounded research Kuramoto HDL emitter configuration."""
        if n_oscillators < 1:
            raise ValueError("n_oscillators must be >= 1")
        if data_width < 16:
            raise ValueError("data_width must be >= 16")
        if not 0 < fraction < data_width:
            raise ValueError("fraction must satisfy 0 < fraction < data_width")
        if lut_size < 16 or lut_size & (lut_size - 1):
            raise ValueError("lut_size must be a power of two >= 16")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if not math.isfinite(coupling):
            raise ValueError("coupling must be finite")

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
        if not all(math.isfinite(omega) for omega in self.omegas):
            raise ValueError("omegas must contain only finite values")
        if not all(math.isfinite(phase) for phase in self.initial_phases):
            raise ValueError("initial_phases must contain only finite values")
        self._validate_fixed_point_format()
        self._validate_single_step_wrap_bound()

    def _fixed_int(self, value: float) -> int:
        """Quantize a real-valued scalar into the configured fixed-point format."""
        return int(round(value * (1 << self.fraction)))

    def _phase_modulus_fixed(self) -> int:
        """Return the fixed-point representation of the ``2pi`` phase modulus."""
        return self._fixed_int(2.0 * math.pi)

    def _require_representable_fixed(self, value: float, name: str) -> None:
        """Reject fixed-point constants that exceed the signed data-path range."""
        fixed = self._fixed_int(value)
        min_signed = -(1 << (self.data_width - 1))
        max_signed = (1 << (self.data_width - 1)) - 1
        if fixed < min_signed or fixed > max_signed:
            raise ValueError(
                f"{name} fixed-point value {fixed} exceeds signed Q{self.data_width - self.fraction}."
                f"{self.fraction} range [{min_signed}, {max_signed}]"
            )

    def _validate_fixed_point_format(self) -> None:
        """Validate that constants and configured inputs fit the fixed-point format."""
        try:
            self._require_representable_fixed(2.0 * math.pi, "phase modulus")
        except ValueError as exc:
            raise ValueError("fixed-point format cannot represent 2pi") from exc
        self._require_representable_fixed(math.pi, "half phase modulus")
        self._require_representable_fixed(self.dt, "dt")
        self._require_representable_fixed(self.coupling / self.n_oscillators, "coupling / N")
        for idx, omega in enumerate(self.omegas):
            self._require_representable_fixed(omega, f"omega[{idx}]")
        for idx, phase in enumerate(self.initial_phases):
            self._require_representable_fixed(phase % (2.0 * math.pi), f"initial_phases[{idx}]")

    def _validate_single_step_wrap_bound(self) -> None:
        """Reject configurations that could require multiple wraps in one RTL step."""
        max_omega = max(abs(omega) for omega in self.omegas)
        max_coupling_term = abs(self.coupling) * max(0, self.n_oscillators - 1) / self.n_oscillators
        max_phase_advance = self.dt * (max_omega + max_coupling_term)
        if max_phase_advance >= 2.0 * math.pi:
            raise ValueError("single-step phase advance must stay below 2pi")

    def _signed_literal(self, value: int) -> str:
        """Format a signed fixed-point integer as a Verilog decimal literal."""
        magnitude = abs(value)
        if value < 0:
            return f"-{self.data_width}'sd{magnitude}"
        return f"{self.data_width}'sd{magnitude}"

    def initial_phase_state_fixed(self) -> list[int]:
        """Return the emitted fixed-point reset state for each oscillator."""
        return [self._fixed_int(phase % (2.0 * math.pi)) for phase in self.initial_phases]

    def fixed_point_step(self, phase_state: list[int] | tuple[int, ...]) -> list[int]:
        """Mirror one generated RTL phase step in integer fixed-point arithmetic."""
        phases = self._validate_phase_state(phase_state)
        phase_modulus = self._phase_modulus_fixed()
        half_phase_modulus = self._fixed_int(math.pi)
        dt_fixed = self._fixed_int(self.dt)
        coupling_fixed = self._fixed_int(self.coupling / self.n_oscillators)
        omega_fixed = [self._fixed_int(omega) for omega in self.omegas]
        next_phases: list[int] = []

        for row, row_phase in enumerate(phases):
            coupling_sum = 0
            for col, col_phase in enumerate(phases):
                if col == row:
                    continue
                phase_diff = self._wrap_delta_fixed(
                    col_phase - row_phase,
                    phase_modulus=phase_modulus,
                    half_phase_modulus=half_phase_modulus,
                )
                coupling_sum += self._sin_lut_fixed(phase_diff, phase_modulus=phase_modulus)
            coupling_term = (coupling_sum * coupling_fixed) >> self.fraction
            phase_velocity = omega_fixed[row] + coupling_term
            phase_delta = (phase_velocity * dt_fixed) >> self.fraction
            next_phases.append(self._wrap_phase_fixed(row_phase + phase_delta, phase_modulus))

        return next_phases

    def fixed_point_error_summary(self, *, steps: int) -> dict[str, int | float | list[float]]:
        """Characterise fixed-point drift against the float Kuramoto Euler step."""
        if steps < 1:
            raise ValueError("steps must be >= 1")

        fixed_state = self.initial_phase_state_fixed()
        float_state = [phase % (2.0 * math.pi) for phase in self.initial_phases]
        max_abs_error = 0.0
        sum_sq_error = 0.0
        sample_count = 0

        for _ in range(steps):
            fixed_state = self.fixed_point_step(fixed_state)
            float_state = self._float_step(float_state)
            fixed_float = self.fixed_state_to_float(fixed_state)
            errors = [
                abs(self._circular_phase_error(fixed_phase, float_phase))
                for fixed_phase, float_phase in zip(fixed_float, float_state)
            ]
            max_abs_error = max(max_abs_error, *errors)
            sum_sq_error += sum(error * error for error in errors)
            sample_count += len(errors)

        rms_error = math.sqrt(sum_sq_error / sample_count)
        return {
            "steps": steps,
            "oscillator_count": self.n_oscillators,
            "data_width": self.data_width,
            "fraction": self.fraction,
            "lut_size": self.lut_size,
            "dt": self.dt,
            "coupling": self.coupling,
            "max_abs_phase_error_rad": max_abs_error,
            "rms_phase_error_rad": rms_error,
            "final_fixed_phases_rad": self.fixed_state_to_float(fixed_state),
            "final_float_phases_rad": float_state,
        }

    def fixed_state_to_float(self, phase_state: list[int] | tuple[int, ...]) -> list[float]:
        """Convert integer fixed-point phase state to radians."""
        phases = self._validate_phase_state(phase_state)
        scale = float(1 << self.fraction)
        return [phase / scale for phase in phases]

    def _validate_phase_state(self, phase_state: list[int] | tuple[int, ...]) -> list[int]:
        """Return a canonical fixed-point phase vector for public mirror helpers."""
        if len(phase_state) != self.n_oscillators:
            raise ValueError("phase_state length must equal n_oscillators")
        phase_modulus = self._phase_modulus_fixed()
        phases: list[int] = []
        for phase in phase_state:
            if not isinstance(phase, int) or isinstance(phase, bool):
                raise ValueError("phase_state entries must be integers")
            if phase < 0 or phase >= phase_modulus:
                raise ValueError("phase_state entries must satisfy 0 <= phase < phase modulus")
            phases.append(phase)
        return phases

    def _float_step(self, phases: list[float]) -> list[float]:
        """Advance the bounded noiseless Kuramoto system with float Euler arithmetic."""
        next_phases: list[float] = []
        for row, row_phase in enumerate(phases):
            coupling_sum = 0.0
            for col, col_phase in enumerate(phases):
                if col == row:
                    continue
                coupling_sum += math.sin(col_phase - row_phase)
            velocity = self.omegas[row] + (self.coupling * coupling_sum / self.n_oscillators)
            next_phases.append((row_phase + self.dt * velocity) % (2.0 * math.pi))
        return next_phases

    @staticmethod
    def _circular_phase_error(actual: float, expected: float) -> float:
        """Return signed circular phase error in ``[-pi, pi)`` radians."""
        return ((actual - expected + math.pi) % (2.0 * math.pi)) - math.pi

    @staticmethod
    def _wrap_phase_fixed(phase_value: int, phase_modulus: int) -> int:
        """Wrap a fixed-point phase into the canonical ``[0, phase_modulus)`` range."""
        return phase_value % phase_modulus

    @staticmethod
    def _wrap_delta_fixed(delta_value: int, *, phase_modulus: int, half_phase_modulus: int) -> int:
        """Wrap a phase difference into the generated RTL's one-step signed range."""
        if delta_value > half_phase_modulus:
            return delta_value - phase_modulus
        if delta_value < -half_phase_modulus:
            return delta_value + phase_modulus
        return delta_value

    def _sin_lut_fixed(self, phase_value: int, *, phase_modulus: int) -> int:
        """Evaluate the same fixed-point sine lookup used by emitted RTL."""
        wrapped_phase = self._wrap_phase_fixed(phase_value, phase_modulus)
        lut_index = (wrapped_phase * self.lut_size) // phase_modulus
        return self._fixed_int(math.sin((2.0 * math.pi * lut_index) / self.lut_size))

    def _lut_lines(self) -> list[str]:
        """Render the Verilog sine lookup table with width-safe case labels."""
        index_width = max(1, math.ceil(math.log2(self.lut_size)))
        lines = [
            "    function automatic signed [DATA_WIDTH-1:0] sin_lut;",
            "        input signed [DATA_WIDTH-1:0] phase_value;",
            "        reg signed [DATA_WIDTH-1:0] wrapped_phase;",
            f"        reg [{index_width - 1}:0] lut_index;",
            "        begin",
            "            wrapped_phase = wrap_phase(phase_value);",
            "            lut_index = (wrapped_phase * LUT_SIZE) / PHASE_MODULUS;",
            "            case (lut_index)",
        ]
        for idx in range(self.lut_size):
            value = self._fixed_int(math.sin((2.0 * math.pi * idx) / self.lut_size))
            lines.append(
                f"                {index_width}'d{idx}: sin_lut = {self._signed_literal(value)};"
            )
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
        """Emit deterministic Verilog for the configured research Kuramoto core."""
        phase_modulus = self._phase_modulus_fixed()
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
