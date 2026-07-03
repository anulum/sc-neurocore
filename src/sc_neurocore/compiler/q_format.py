# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fixed-point format specifications

"""Fixed-point Q-format specifications for weight quantization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

RoundingMode = Literal["nearest", "stochastic", "floor"]
_ROUNDING_MODES: set[str] = {"nearest", "stochastic", "floor"}


@dataclass(frozen=True)
class QFormat:
    """Signed fixed-point Q-format specification.

    Attributes
    ----------
    integer_bits:
        Number of signed integer bits, including the sign bit.
    fraction_bits:
        Number of fractional bits below the binary point.
    """

    integer_bits: int
    fraction_bits: int

    def __post_init__(self) -> None:
        """Validate the fixed-point width fields after construction."""
        if type(self.integer_bits) is not int:
            raise TypeError("integer_bits must be an integer")
        if type(self.fraction_bits) is not int:
            raise TypeError("fraction_bits must be an integer")
        if self.integer_bits < 1:
            raise ValueError("integer_bits must include at least the sign bit")
        if self.fraction_bits < 0:
            raise ValueError("fraction_bits must be non-negative")

    @property
    def total_bits(self) -> int:
        """Total signed fixed-point storage width in bits."""
        return self.integer_bits + self.fraction_bits

    @property
    def scale(self) -> int:
        """Integer scale factor used to encode real values."""
        return 1 << self.fraction_bits

    @property
    def min_val(self) -> float:
        """Smallest representable signed fixed-point value."""
        return -(1 << (self.total_bits - 1)) / self.scale

    @property
    def max_val(self) -> float:
        """Largest representable signed fixed-point value."""
        return ((1 << (self.total_bits - 1)) - 1) / self.scale

    @property
    def min_value(self) -> float:
        """Minimum representable fixed-point value."""
        return self.min_val

    @property
    def max_value(self) -> float:
        """Maximum representable fixed-point value."""
        return self.max_val

    @property
    def q_label(self) -> str:
        """Canonical Q-format label."""
        return f"Q{self.integer_bits}.{self.fraction_bits}"

    @classmethod
    def from_string(cls, fmt: str) -> QFormat:
        """Parse 'Q8.8', 'Q4.12', etc."""
        if not isinstance(fmt, str):
            raise TypeError(f"Expected Q-format string, got {type(fmt)!r}")

        text = fmt.strip().upper()
        match = re.fullmatch(r"Q(?P<int>\d+)\.(?P<frac>\d+)", text)
        if match is None:
            raise ValueError(f"Expected format like 'Q8.8', got {fmt!r}")
        return cls(
            integer_bits=int(match.group("int")),
            fraction_bits=int(match.group("frac")),
        )


Q8_8 = QFormat(8, 8)
Q16_16 = QFormat(16, 16)


@dataclass(frozen=True)
class QFormatMixed:
    """Mixed fixed-point contract for Q-format weights and wider accumulators.

    Attributes
    ----------
    weight_fmt:
        Stored weight format.
    accum_fmt:
        Wider accumulator format used by mixed-precision dense kernels.
    scale_per_tensor:
        Whether one scale is shared by the tensor instead of per-channel scale.
    rounding:
        Rounding policy applied by the quantizer.
    """

    weight_fmt: QFormat = Q8_8
    accum_fmt: QFormat = Q16_16
    scale_per_tensor: bool = True
    rounding: RoundingMode = "nearest"

    def __post_init__(self) -> None:
        """Validate mixed-format compatibility after construction."""
        if not isinstance(self.weight_fmt, QFormat):
            raise TypeError("weight_fmt must be a QFormat")
        if not isinstance(self.accum_fmt, QFormat):
            raise TypeError("accum_fmt must be a QFormat")
        if type(self.scale_per_tensor) is not bool:
            raise TypeError("scale_per_tensor must be a boolean")
        if self.rounding not in _ROUNDING_MODES:
            raise ValueError("rounding must be 'nearest', 'stochastic', or 'floor'")
        if self.accum_fmt.total_bits < self.weight_fmt.total_bits:
            raise ValueError("accum_fmt must be at least as wide as weight_fmt")
        if self.accum_fmt.fraction_bits < self.weight_fmt.fraction_bits:
            raise ValueError("accum_fmt must preserve at least the weight fractional precision")
        if (
            self.accum_fmt.min_value > self.weight_fmt.min_value
            or self.accum_fmt.max_value < self.weight_fmt.max_value
        ):
            raise ValueError("accum_fmt must cover the full weight dynamic range")

    @property
    def accumulator_guard_bits(self) -> int:
        """Extra accumulator bits available above the stored weight width."""
        return self.accum_fmt.total_bits - self.weight_fmt.total_bits

    @property
    def metadata(self) -> dict[str, bool | int | str]:
        """Deterministic metadata for manifests and hardware telemetry."""
        return {
            "kind": "mixed_fixed_point",
            "weight_format": self.weight_fmt.q_label,
            "accumulator_format": self.accum_fmt.q_label,
            "weight_total_bits": self.weight_fmt.total_bits,
            "accumulator_total_bits": self.accum_fmt.total_bits,
            "accumulator_guard_bits": self.accumulator_guard_bits,
            "scale_per_tensor": self.scale_per_tensor,
            "rounding": self.rounding,
        }
