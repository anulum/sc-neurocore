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
from fractions import Fraction


# Transcendental lowerings: "lut-v1" (fixed tables, the default) and "v2" (function units).
FUNCTION_LOWERINGS = ("lut-v1", "v2")
RANDOM_PRECISIONS = ("standard", "exact")


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
    function_lowering : str
        How transcendental functions are lowered: ``"lut-v1"`` (default; one
        sample of a fixed table) or ``"v2"`` (range reduction and quadratic
        tables, :mod:`sc_neurocore.compiler.function_units`).
    random_precision : str
        How ``randn()`` is lowered: ``"standard"`` (default; 30-bit internals,
        within 0.5 LSB + ``2**-24``) or ``"exact"`` (60-bit internals, within
        0.5 LSB + ``2**-45``, :mod:`sc_neurocore.compiler.exact_normal`).

    """

    data_width: int = 16
    fraction: int = 8
    signed: bool = True
    overflow: str = "saturate"  # saturate | wrap | trap
    rounding: str = "truncate"  # truncate | nearest | bankers | stochastic
    function_lowering: str = "lut-v1"  # lut-v1 | v2
    random_precision: str = "standard"  # standard | exact (randn only)

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
        if self.random_precision not in RANDOM_PRECISIONS:
            raise ValueError(
                f"random_precision must be one of {RANDOM_PRECISIONS}, got {self.random_precision!r}"
            )
        if self.function_lowering not in FUNCTION_LOWERINGS:
            raise ValueError(
                f"function_lowering must be one of {FUNCTION_LOWERINGS}, got {self.function_lowering!r}"
            )

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
        raw = round(Fraction(value) * (1 << self.fraction))  # exact; no float overflow
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
        raw = round(Fraction(value) * (1 << self.fraction))  # exact; no float overflow
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

    def exact_code(self, value: float) -> int:
        """Return ``round(value · 2**fraction)`` exactly, before any range check.

        Parameters
        ----------
        value : float
            A finite value.

        Returns
        -------
        int
            The nearest code (ties to even, as Python's ``round``), unbounded:
            compare it with the word's range to see whether it fits.
        """
        return round(Fraction(value) * (1 << self.fraction))

    def require_in_range(self, value: float, label: str) -> None:
        """Refuse a value outside the format, with the :meth:`check_range` message.

        Raises
        ------
        ValueError
            If ``value`` exceeds the format's range.
        """
        problems = self.check_range(value, label)
        if problems:
            raise ValueError(problems[0])

    def check_time_step(self, dt: float, method: str) -> None:
        """Refuse a dt the datapath multiplies by but the format cannot hold.

        A map (``method="map"``) never multiplies by dt; every other method does,
        and a wrapped dt (1.0 at Q1.15 is -1.0) would integrate a different step.

        Raises
        ------
        ValueError
            If ``method`` uses dt and ``dt`` is outside the format.
        """
        if method != "map":
            self.require_in_range(dt, "dt")

    def _relative_error_percent(self, code: int, value: float) -> float:
        """Return ``|code/2**fraction - value| / |value|`` in percent, exactly then rounded."""
        exact = Fraction(value)
        return float(abs(Fraction(code, 1 << self.fraction) - exact) / abs(exact) * 100)

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

        # dt analysis, in exact arithmetic: a huge finite dt has an exact code
        # (a host-float product would overflow) and is reported out of range.
        dt_raw = self.exact_code(dt)
        dt_actual = float(Fraction(dt_raw, 1 << self.fraction))
        dt_error = (
            self._relative_error_percent(dt_raw, dt)
            if dt != 0 and dt_raw != 0
            else (100.0 if dt != 0 else 0.0)
        )
        if dt == 0.0:
            dt_status = "✓ ZERO STEP"
        elif self.check_range(dt, "dt"):
            dt_status = "✗ OUT OF RANGE"
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
                q_val = self.exact_code(val)
                err = self._relative_error_percent(q_val, val) if val != 0 else 0
                lines.append(f"  {name}={val} → Q-value={q_val} (error={err:.1f}%)")

            for warning in range_warnings:
                lines.append(f"  ⚠ {warning}")

        return "\n".join(lines)
