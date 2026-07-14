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
    """Describe a fixed-point word used by compiler diagnostics and emitters.

    The historical class name is retained for API compatibility, but
    ``data_width`` and ``fraction`` are configurable. Unsigned instances are
    valid for range analysis and raw-word encoding. The equation-to-Verilog
    emitters currently reject unsigned instances because their state and
    expression datapaths are signed.

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
    - ``"saturate"`` — clamp to the representable interval (default)
    - ``"wrap"``     — two's complement wrap-around (Loihi 2 hardware behaviour)
    - ``"trap"``     — emit a simulation-only ``$fatal`` assertion

    Rounding Modes
    ~~~~~~~~~~~~~~
    - ``"truncate"``   — arithmetic shift towards negative infinity (default)
    - ``"nearest"``    — round to nearest, ties away from zero
    - ``"bankers"``    — round to nearest, ties to even (IEEE 754 default)
    - ``"stochastic"`` — reserved label; equation-to-Verilog emission rejects it

    Parameters
    ----------
    data_width : int
        Total number of bits in the encoded word.
    fraction : int
        Number of fractional bits.
    signed : bool
        Whether range diagnostics interpret the word as two's-complement.
    overflow : str
        Overflow policy consumed by the Verilog emitters.
    rounding : str
        Product-rounding policy consumed by the expression emitter. The public
        equation-to-Verilog paths reject ``"stochastic"`` because they do not
        own a rounding LFSR.
    """

    data_width: int = 16
    fraction: int = 8
    signed: bool = True
    overflow: str = "saturate"  # saturate | wrap | trap
    rounding: str = "truncate"  # truncate | nearest | bankers | stochastic

    def __post_init__(self) -> None:
        """Validate the fixed-point geometry and declared arithmetic modes.

        Raises
        ------
        TypeError
            If a geometry or mode field has the wrong runtime type.
        ValueError
            If the word geometry is impossible.
        """
        if type(self.data_width) is not int:
            raise TypeError(f"data_width must be an integer, got {self.data_width!r}")
        if self.data_width < 1:
            raise ValueError(f"data_width must be positive, got {self.data_width}")
        if type(self.fraction) is not int:
            raise TypeError(f"fraction must be an integer, got {self.fraction!r}")
        if type(self.signed) is not bool:
            raise TypeError(f"signed must be a boolean, got {self.signed!r}")
        if not 0 <= self.fraction <= self.data_width:
            raise ValueError(
                f"fraction must satisfy 0 <= fraction <= {self.data_width}; got {self.fraction}"
            )
        if type(self.overflow) is not str:
            raise TypeError(f"overflow must be a string, got {self.overflow!r}")
        if type(self.rounding) is not str:
            raise TypeError(f"rounding must be a string, got {self.rounding!r}")

    @property
    def integer_bits(self) -> int:
        """Return the number of magnitude bits above the binary point."""
        return max(0, self.data_width - self.fraction - (1 if self.signed else 0))

    @property
    def max_value(self) -> float:
        """Return the largest representable value."""
        if self.signed:
            return ((1 << (self.data_width - 1)) - 1) / (1 << self.fraction)
        return ((1 << self.data_width) - 1) / (1 << self.fraction)

    @property
    def min_value(self) -> float:
        """Return the smallest representable value."""
        if self.signed:
            return -(1 << (self.data_width - 1)) / (1 << self.fraction)
        return 0.0

    @property
    def resolution(self) -> float:
        """Return the spacing between adjacent encoded values."""
        return 1.0 / (1 << self.fraction)

    def encode(self, value: float) -> int:
        """Encode a value as its fixed-width raw bit pattern.

        Parameters
        ----------
        value : float
            The value to encode (e.g. -65.0 for a membrane voltage).

        Returns
        -------
        int
            The encoded word masked to ``data_width`` bits. For a signed
            format, negative inputs use two's-complement representation.

        Notes
        -----
        This method does not apply the configured overflow policy. Call
        :meth:`check_range` before encoding externally supplied values.
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

        Raises
        ------
        ValueError
            If this configuration is unsigned and therefore cannot be
            represented by a signed-literal contract.
        """
        if not self.signed:
            raise ValueError("signed Verilog literals require signed=True")
        raw = int(round(value * (1 << self.fraction)))
        if raw < 0:
            raw = raw & ((1 << self.data_width) - 1)
        return f"{self.data_width}'sd{raw}"

    def check_range(self, value: float, label: str = "") -> list[str]:
        """Return diagnostic messages when a value is outside the format.

        Parameters
        ----------
        value : float
            Value to compare with the representable interval.
        label : str
            Optional name included in diagnostic messages.

        Returns
        -------
        list[str]
            An empty list for an in-range value, otherwise one overflow or
            underflow message.
        """
        warnings: list[str] = []
        prefix = "Q" if self.signed else "UQ"
        if value > self.max_value:
            warnings.append(
                f"Overflow: {label}={value} exceeds "
                f"{prefix}{self.data_width - self.fraction}.{self.fraction} "
                f"max={self.max_value:.4f}"
            )
        elif value < self.min_value:
            warnings.append(
                f"Underflow: {label}={value} below "
                f"{prefix}{self.data_width - self.fraction}.{self.fraction} "
                f"min={self.min_value:.4f}"
            )
        return warnings

    def precision_report(self, dt: float, params: dict[str, float] | None = None) -> str:
        """Build a fixed-point quantisation diagnostics report.

        Parameters
        ----------
        dt : float
            Integration step to quantise in the configured format.
        params : dict[str, float], optional
            Named values to quantise and range-check.

        Returns
        -------
        str
            Multi-line format, timestep, and parameter diagnostics.
        """
        prefix = "Q" if self.signed else "UQ"
        signedness = "signed" if self.signed else "unsigned"
        lines = [
            f"Fixed-point format: {prefix}{self.data_width - self.fraction}.{self.fraction} "
            f"({self.data_width}-bit {signedness})",
            f"  Integer range: [{self.min_value:.4f}, {self.max_value:.4f}]",
            f"  Resolution: {self.resolution:.6f}",
        ]

        # dt analysis
        dt_raw = int(round(dt * (1 << self.fraction)))
        dt_actual = dt_raw / (1 << self.fraction) if dt_raw != 0 else 0.0
        dt_error = (
            abs(dt_actual - dt) / abs(dt) * 100
            if dt != 0 and dt_raw != 0
            else (100.0 if dt != 0 else 0.0)
        )
        if dt == 0.0:
            dt_status = "✓ ZERO STEP"
        elif dt_raw == 0:
            dt_status = "✗ UNDERFLOW"
        else:
            dt_status = "✓"
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
