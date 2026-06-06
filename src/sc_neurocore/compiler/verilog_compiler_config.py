# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog compiler configuration

"""Fixed-point configuration for Verilog code generation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Q88:
    """Fixed-point format configuration for Verilog code generation.

    Supports arbitrary precision modes via ``data_width`` / ``fraction``,
    with configurable signedness, overflow handling, and rounding.

    ============  ==========  ===============  =================  ===============
    Mode          data_width  fraction         Integer range      Resolution
    ============  ==========  ===============  =================  ===============
    **Q8.8**      16          8                [-128, +127.996]   1/256 ≈ 0.004
    **Q4.12**     16          12               [-8, +7.9998]      1/4096 ≈ 0.0002
    **Q16.16**    32          16               [-32768, +32767]   1/65536 ≈ 1.5e-5
    **UQ8.8**     16          8  (unsigned)    [0, +255.996]      1/256 ≈ 0.004
    ============  ==========  ===============  =================  ===============

    Overflow Modes
    ~~~~~~~~~~~~~~
    - ``"saturate"`` — clamp to [min, max] (default, safest)
    - ``"wrap"``     — two's complement wrap-around (Loihi 2 hardware behaviour)
    - ``"trap"``     — emit ``$fatal`` assertion (DO-254 / IEC 61508 safety)

    Rounding Modes
    ~~~~~~~~~~~~~~
    - ``"truncate"``   — floor towards -∞ (default, fastest)
    - ``"nearest"``    — round to nearest, ties away from zero
    - ``"bankers"``    — round to nearest, ties to even (IEEE 754 default)
    - ``"stochastic"`` — probabilistic rounding via LFSR (reduces long-run bias)
    """

    data_width: int = 16
    fraction: int = 8
    signed: bool = True
    overflow: str = "saturate"  # saturate | wrap | trap
    rounding: str = "truncate"  # truncate | nearest | bankers | stochastic

    @property
    def integer_bits(self) -> int:
        """Number of integer bits (excluding sign bit if signed)."""
        return self.data_width - self.fraction - (1 if self.signed else 0)

    @property
    def max_value(self) -> float:
        """Maximum representable positive value."""
        if self.signed:
            return ((1 << (self.data_width - 1)) - 1) / (1 << self.fraction)
        return ((1 << self.data_width) - 1) / (1 << self.fraction)

    @property
    def min_value(self) -> float:
        """Minimum (most negative) representable value."""
        if self.signed:
            return -(1 << (self.data_width - 1)) / (1 << self.fraction)
        return 0.0

    @property
    def resolution(self) -> float:
        """Smallest representable non-zero step."""
        return 1.0 / (1 << self.fraction)

    def encode(self, value: float) -> int:
        """Encode a float to unsigned Q-format integer representation.

        Parameters
        ----------
        value : float
            The value to encode (e.g. -65.0 for a membrane voltage).

        Returns
        -------
        int
            The unsigned two's complement representation, masked to
            ``data_width`` bits.  Example: -65.0 in Q8.8 → 48896.
        """
        raw = int(round(value * (1 << self.fraction)))
        mask = (1 << self.data_width) - 1
        return raw & mask

    def encode_signed_literal(self, value: float) -> str:
        """Encode a float as a Verilog signed decimal literal.

        Produces a string like ``16'sd48896`` (Q8.8) or ``32'sd4259840``
        (Q16.16) suitable for embedding directly in generated Verilog.

        Parameters
        ----------
        value : float
            The value to encode.

        Returns
        -------
        str
            Verilog literal, e.g. ``"16'sd48896"`` for -65.0 in Q8.8.
        """
        raw = int(round(value * (1 << self.fraction)))
        if raw < 0:
            raw = raw & ((1 << self.data_width) - 1)
        return f"{self.data_width}'sd{raw}"

    def check_range(self, value: float, label: str = "") -> list[str]:
        """Check if a value fits in the integer range. Returns warnings."""
        warnings = []
        if value > self.max_value:
            warnings.append(
                f"Overflow: {label}={value} exceeds Q{self.data_width - self.fraction}.{self.fraction} "
                f"max={self.max_value:.4f}"
            )
        elif value < self.min_value:
            warnings.append(
                f"Underflow: {label}={value} below Q{self.data_width - self.fraction}.{self.fraction} "
                f"min={self.min_value:.4f}"
            )
        return warnings

    def precision_report(self, dt: float, params: dict[str, float] | None = None) -> str:
        """Generate a human-readable precision diagnostics report."""
        lines = [
            f"Fixed-point format: Q{self.data_width - self.fraction}.{self.fraction} "
            f"({self.data_width}-bit signed)",
            f"  Integer range: [{self.min_value:.4f}, {self.max_value:.4f}]",
            f"  Resolution: {self.resolution:.6f}",
        ]

        # dt analysis
        dt_raw = int(round(dt * (1 << self.fraction)))
        dt_actual = dt_raw / (1 << self.fraction) if dt_raw != 0 else 0.0
        dt_error = (
            abs(dt_actual - dt) / dt * 100
            if dt != 0 and dt_raw != 0
            else (100.0 if dt != 0 else 0.0)
        )
        dt_status = "✓" if dt_raw > 0 else "✗ UNDERFLOW"
        lines.append(
            f"  dt={dt} → Q-value={dt_raw} (actual={dt_actual:.6f}, "
            f"error={dt_error:.1f}%) {dt_status}"
        )

        # Parameter analysis
        if params:
            range_warnings: list[str] = []
            for name, val in params.items():
                warnings_for_param = self.check_range(val, name)
                range_warnings.extend(warnings_for_param)
                q_val = int(round(val * (1 << self.fraction)))
                q_actual = q_val / (1 << self.fraction)
                err = abs(q_actual - val) / abs(val) * 100 if val != 0 else 0
                lines.append(f"  {name}={val} → Q-value={q_val} (error={err:.1f}%)")

            for warning in range_warnings:
                lines.append(f"  ⚠ {warning}")

        return "\n".join(lines)
