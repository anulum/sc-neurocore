# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Weight quantization for hardware deployment

"""Quantize trained float weights to fixed-point / block-floating precision."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

RoundingMode = Literal["nearest", "stochastic", "floor"]
_ROUNDING_MODES: set[str] = {"nearest", "stochastic", "floor"}


@dataclass(frozen=True)
class QFormat:
    """Fixed-point Q-format specification."""

    integer_bits: int
    fraction_bits: int

    def __post_init__(self) -> None:
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
        return self.integer_bits + self.fraction_bits

    @property
    def scale(self) -> int:
        return 1 << self.fraction_bits

    @property
    def min_val(self) -> float:
        return -(1 << (self.total_bits - 1)) / self.scale

    @property
    def max_val(self) -> float:
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
    """Mixed fixed-point contract for Q-format weights and wider accumulators."""

    weight_fmt: QFormat = Q8_8
    accum_fmt: QFormat = Q16_16
    scale_per_tensor: bool = True
    rounding: RoundingMode = "nearest"

    def __post_init__(self) -> None:
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


@dataclass(frozen=True)
class BlockFloatingMode:
    """Shared-exponent block floating-point specification."""

    mantissa_bits: int
    exponent_bits: int
    block_size: int = 32

    @property
    def label(self) -> str:
        """Human-readable label."""
        return f"BFP{self.mantissa_bits}E{self.exponent_bits}"

    @property
    def exponent_bias(self) -> int:
        """Bias applied to shared exponents."""
        return (1 << (self.exponent_bits - 1)) - 1

    @property
    def min_exponent(self) -> int:
        """Minimum unbiased exponent."""
        return -self.exponent_bias

    @property
    def max_exponent(self) -> int:
        """Maximum unbiased exponent."""
        return ((1 << self.exponent_bits) - 1) - self.exponent_bias

    @property
    def mantissa_range(self) -> int:
        """Largest signed mantissa magnitude."""
        return (1 << (self.mantissa_bits - 1)) - 1

    @property
    def emit_fraction(self) -> int:
        """Conservative fixed-point fallback fraction for RTL emission."""
        return max(1, self.mantissa_bits - 1)

    @property
    def metadata(self) -> dict[str, int | str]:
        """Deterministic metadata payload for cross-target telemetry."""
        return {
            "kind": "block_floating",
            "mantissa_bits": self.mantissa_bits,
            "exponent_bits": self.exponent_bits,
            "block_size": self.block_size,
            "exponent_min": self.min_exponent,
            "exponent_max": self.max_exponent,
            "block_exponent_alignment": "contiguous_flattened_block",
            "block_exponent_count_policy": "ceil(parameter_count / block_size)",
        }

    def block_exponent_count(self, parameter_count: int) -> int:
        """Return the exact number of shared exponents for a flat parameter payload."""
        if type(parameter_count) is not int:
            raise TypeError("parameter_count must be an integer")
        if parameter_count < 0:
            raise ValueError("parameter_count must be non-negative")
        if parameter_count == 0:
            return 0
        return (parameter_count + self.block_size - 1) // self.block_size

    def block_exponent_layout(self, parameter_count: int) -> "BlockExponentLayout":
        """Return the explicit exponent-vector layout for downstream emitters."""
        return BlockExponentLayout(
            parameter_count=parameter_count,
            block_size=self.block_size,
            exponent_count=self.block_exponent_count(parameter_count),
        )

    def validate_exponents(
        self,
        exponents: np.ndarray[Any, Any],
        *,
        parameter_count: int,
    ) -> np.ndarray[Any, Any]:
        """Validate exponent vector length and code range for a parameter payload."""
        layout = self.block_exponent_layout(parameter_count)
        return layout.validate_exponents(exponents, exponent_bits=self.exponent_bits)

    @classmethod
    def from_string(cls, fmt: str) -> "BlockFloatingMode":
        """Parse strict canonical format like 'BFP16E3'."""
        text = fmt.strip().upper()
        if not text.startswith("BFP"):
            raise ValueError(f"Expected block-floating format like 'BFP16E3', got {fmt!r}")

        body = text[3:]
        if not body:
            raise ValueError(f"Expected block-floating format like 'BFP16E3', got {fmt!r}")

        m = re.fullmatch(r"(?P<left>\d+)E(?P<right>\d+)(?:X(?P<block>\d+))?$", body)
        if not m:
            raise ValueError(f"Expected block-floating format like 'BFP16E3', got {fmt!r}")

        mantissa_bits = int(m.group("left"))
        exponent_bits = int(m.group("right"))
        block_size = int(m.group("block") or 32)

        if mantissa_bits < 2:
            raise ValueError(f"Mantissa bits must be at least 2, got {mantissa_bits}")
        if exponent_bits < 1:
            raise ValueError(f"Exponent bits must be at least 1, got {exponent_bits}")
        if block_size < 1:
            raise ValueError(f"Block size must be positive, got {block_size}")

        return cls(mantissa_bits=mantissa_bits, exponent_bits=exponent_bits, block_size=block_size)

    @classmethod
    def from_aliases(cls, fmt: str) -> "BlockFloatingMode":
        """Parse tolerant aliases such as BFP16_E3, BFP16.3, BFP16-3, and BFP16E3X32."""
        if not isinstance(fmt, str):
            raise TypeError(f"Expected BFP format string, got {type(fmt)!r}")

        text = fmt.strip().upper()
        if not text.startswith("BFP"):
            raise ValueError(f"Expected block-floating format like 'BFP16E3', got {fmt!r}")

        body = text[3:]
        if "E" not in body:
            body = re.sub(r"(?<=\d)[._-](?=\d)", "E", body)

        parts = body.split("X")
        if len(parts) > 2:
            raise ValueError(f"Expected block-floating format like 'BFP16E3', got {fmt!r}")

        core = re.sub(r"[._-](?=\d)", "E", parts[0])
        core = re.sub(r"[._-]", "", core)
        if "E" not in core:
            raise ValueError(f"Expected block-floating format like 'BFP16E3', got {fmt!r}")

        if len(parts) == 1:
            return cls.from_string(f"BFP{core}")

        if not parts[1] or not parts[1].isdigit():
            raise ValueError(f"Expected block size in 'BFP16E3X32', got {fmt!r}")

        return cls.from_string(f"BFP{core}X{int(parts[1])}")


def parse_precision_format(fmt: str) -> QFormat | BlockFloatingMode:
    """Parse fixed-point or block-floating precision format."""
    if not isinstance(fmt, str):
        raise TypeError(f"Expected precision format string, got {type(fmt)!r}")

    try:
        return BlockFloatingMode.from_aliases(fmt)
    except (ValueError, TypeError):
        return QFormat.from_string(fmt)


@dataclass(frozen=True)
class BlockExponentLayout:
    """Concrete shared-exponent layout for flattened block-floating parameters."""

    parameter_count: int
    block_size: int
    exponent_count: int
    alignment: str = "contiguous_flattened_block"
    flattened_order: str = "row_major"

    def __post_init__(self) -> None:
        if type(self.parameter_count) is not int:
            raise TypeError("parameter_count must be an integer")
        if type(self.block_size) is not int:
            raise TypeError("block_size must be an integer")
        if type(self.exponent_count) is not int:
            raise TypeError("exponent_count must be an integer")
        if self.parameter_count < 0:
            raise ValueError("parameter_count must be non-negative")
        if self.block_size < 1:
            raise ValueError("block_size must be positive")
        expected = 0
        if self.parameter_count:
            expected = (self.parameter_count + self.block_size - 1) // self.block_size
        if self.exponent_count != expected:
            raise ValueError(
                f"exponent_count mismatch: expected {expected}, got {self.exponent_count}"
            )
        if self.alignment != "contiguous_flattened_block":
            raise ValueError("alignment must be contiguous_flattened_block")
        if self.flattened_order != "row_major":
            raise ValueError("flattened_order must be row_major")

    @property
    def last_block_size(self) -> int:
        """Number of parameters carried by the final exponent block."""
        if self.parameter_count == 0:
            return 0
        remainder = self.parameter_count % self.block_size
        return remainder or self.block_size

    def manifest(self) -> dict[str, int | str]:
        """Deterministic block-exponent layout manifest."""
        return {
            "alignment": self.alignment,
            "flattened_order": self.flattened_order,
            "parameter_count": self.parameter_count,
            "block_size": self.block_size,
            "exponent_count": self.exponent_count,
            "last_block_size": self.last_block_size,
            "exponent_index_formula": "parameter_index // block_size",
        }

    def validate_exponents(
        self,
        exponents: np.ndarray[Any, Any],
        *,
        exponent_bits: int,
    ) -> np.ndarray[Any, Any]:
        """Validate exponent vector length and encoded range."""
        raw = np.asarray(exponents)
        if not np.issubdtype(raw.dtype, np.integer):
            raise TypeError("exponents must contain integer codes")
        codes = raw.astype(np.int64, copy=True).reshape(-1)
        if codes.size != self.exponent_count:
            raise ValueError(
                f"exponent count mismatch: expected {self.exponent_count}, got {codes.size}"
            )
        max_code = (1 << exponent_bits) - 1
        if np.any(codes < 0) or np.any(codes > max_code):
            raise ValueError("exponents exceed the configured block-floating range")
        return codes


def _coerce_q_format(fmt: str | QFormat) -> QFormat:
    if isinstance(fmt, QFormat):
        return fmt
    if isinstance(fmt, str):
        return QFormat.from_string(fmt)
    raise TypeError(f"Expected QFormat or Q-format string, got {type(fmt)!r}")


def _fixed_integer_bounds(q: QFormat) -> tuple[int, int]:
    return -(1 << (q.total_bits - 1)), (1 << (q.total_bits - 1)) - 1


def _finite_float_array(values: np.ndarray[Any, Any], *, label: str) -> np.ndarray[Any, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values")
    return arr


def _round_scaled(scaled: np.ndarray[Any, Any], rounding: str) -> np.ndarray[Any, Any]:
    if rounding == "nearest":
        return np.rint(scaled).astype(np.int64)
    if rounding == "stochastic":
        floor = np.floor(scaled)
        probability = scaled - floor
        return (floor + (np.random.random(scaled.shape) < probability)).astype(np.int64)
    if rounding == "floor":
        return np.floor(scaled).astype(np.int64)
    raise ValueError(
        f"Unknown rounding mode: {rounding!r}. Use 'nearest', 'stochastic', or 'floor'."
    )


def _quantize_fixed_array(
    weights: np.ndarray[Any, Any],
    q: QFormat,
    *,
    rounding: str,
    clip: bool,
) -> np.ndarray[Any, Any]:
    w = _finite_float_array(weights, label="weights")

    if clip:
        w = np.clip(w, q.min_value, q.max_value)

    quantized = _round_scaled(w * q.scale, rounding)
    min_int, max_int = _fixed_integer_bounds(q)
    return np.clip(quantized, min_int, max_int)


def _mixed_tensor_scale(weights: np.ndarray[Any, Any], fmt: QFormatMixed) -> float:
    if not fmt.scale_per_tensor or weights.size == 0:
        return 1.0

    max_abs = float(np.max(np.abs(weights)))
    if max_abs == 0.0:
        return 1.0

    _, max_int = _fixed_integer_bounds(fmt.weight_fmt)
    return max_int / (max_abs * fmt.weight_fmt.scale)


def _quantize_mixed_precision_weights(
    weights: np.ndarray[Any, Any],
    fmt: QFormatMixed,
    *,
    rounding: str,
    clip: bool,
) -> tuple[np.ndarray[Any, Any], float]:
    w = _finite_float_array(weights, label="weights")
    tensor_scale = _mixed_tensor_scale(w, fmt)
    if not math.isfinite(tensor_scale) or tensor_scale <= 0.0:
        raise ValueError("per-tensor scale must be finite and positive")

    if clip and not fmt.scale_per_tensor:
        w = np.clip(w, fmt.weight_fmt.min_value, fmt.weight_fmt.max_value)

    quantized = _round_scaled(w * fmt.weight_fmt.scale * tensor_scale, rounding)
    min_int, max_int = _fixed_integer_bounds(fmt.weight_fmt)
    return np.clip(quantized, min_int, max_int), tensor_scale


def quantize_weights(
    weights: np.ndarray[Any, Any],
    fmt: str | QFormat | QFormatMixed = "Q8.8",
    rounding: str | None = None,
    clip: bool = True,
) -> np.ndarray[Any, Any] | tuple[np.ndarray[Any, Any], float]:
    """Quantize float weights to fixed-point integers.

    Parameters
    ----------
    weights : np.ndarray
        Float weight matrix (any shape).
    fmt : str | QFormat | QFormatMixed
        Q-format string/object, e.g. ``"Q8.8"`` or ``QFormatMixed()``.
    rounding : str
        ``nearest`` (round half to even), ``stochastic``, or ``floor``.
    clip : bool
        If True, clip values to the representable range before quantization.
    """
    if isinstance(fmt, QFormatMixed):
        return _quantize_mixed_precision_weights(
            weights,
            fmt,
            rounding=rounding or fmt.rounding,
            clip=clip,
        )

    if isinstance(fmt, str) and fmt.upper().startswith("BFP"):
        raise ValueError(
            "Block-floating formats are supported via quantize_block_floating(); "
            "quantize_weights() is fixed-point only."
        )

    q = _coerce_q_format(fmt)
    return _quantize_fixed_array(weights, q, rounding=rounding or "nearest", clip=clip)


@dataclass(frozen=True)
class PrecisionTrapReport:
    """Per-output fixed-point saturation report for deployment trap wiring."""

    operation: str
    output_codes: np.ndarray[Any, Any]
    overflow_mask: np.ndarray[Any, Any]
    output_fmt: QFormat
    underflow_mask: np.ndarray[Any, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str) or not self.operation:
            raise ValueError("operation must be a non-empty string")
        if not isinstance(self.output_fmt, QFormat):
            raise TypeError("output_fmt must be a QFormat")

        raw_codes = np.asarray(self.output_codes)
        if not np.issubdtype(raw_codes.dtype, np.integer):
            raise TypeError("output_codes must contain integer fixed-point codes")
        codes = raw_codes.astype(np.int64, copy=True)
        mask = np.asarray(self.overflow_mask, dtype=bool)
        if self.underflow_mask is None:
            underflow = np.zeros(mask.shape, dtype=bool)
        else:
            underflow = np.asarray(self.underflow_mask, dtype=bool)

        if codes.ndim != 1:
            raise ValueError("output_codes must be a 1-D vector")
        if mask.ndim != 1:
            raise ValueError("overflow_mask must be a 1-D vector")
        if underflow.ndim != 1:
            raise ValueError("underflow_mask must be a 1-D vector")
        if codes.shape != mask.shape or codes.shape != underflow.shape:
            raise ValueError("output_codes, overflow_mask, and underflow_mask must have identical shape")

        min_code, max_code = _fixed_integer_bounds(self.output_fmt)
        if np.any(codes < min_code) or np.any(codes > max_code):
            raise ValueError("output_codes exceed the configured output format")

        object.__setattr__(self, "output_codes", codes)
        object.__setattr__(self, "overflow_mask", mask.astype(bool, copy=True))
        object.__setattr__(self, "underflow_mask", underflow.astype(bool, copy=True))

    @property
    def output_count(self) -> int:
        """Number of output channels covered by this report."""
        return int(self.output_codes.size)

    @property
    def overflow_count(self) -> int:
        """Number of outputs that saturated during the producing operation."""
        return int(np.count_nonzero(self.overflow_mask))

    @property
    def has_overflow(self) -> bool:
        """Whether any output channel saturated."""
        return self.overflow_count > 0

    @property
    def underflow_count(self) -> int:
        """Number of nonzero outputs that collapsed below one output LSB."""
        return int(np.count_nonzero(self.underflow_mask))

    @property
    def has_underflow(self) -> bool:
        """Whether any nonzero output collapsed to the zero code."""
        return self.underflow_count > 0

    @property
    def saturated_min_count(self) -> int:
        """Number of outputs clamped to the minimum representable code."""
        min_code, _ = _fixed_integer_bounds(self.output_fmt)
        return int(np.count_nonzero(self.output_codes == min_code))

    @property
    def saturated_max_count(self) -> int:
        """Number of outputs clamped to the maximum representable code."""
        _, max_code = _fixed_integer_bounds(self.output_fmt)
        return int(np.count_nonzero(self.output_codes == max_code))

    def manifest(self) -> dict[str, bool | int | str]:
        """Deterministic trap metadata for host and hardware telemetry."""
        return {
            "operation": self.operation,
            "output_format": self.output_fmt.q_label,
            "output_count": self.output_count,
            "overflow_count": self.overflow_count,
            "underflow_count": self.underflow_count,
            "saturated_min_count": self.saturated_min_count,
            "saturated_max_count": self.saturated_max_count,
            "has_overflow": self.has_overflow,
            "has_underflow": self.has_underflow,
        }


@dataclass(frozen=True)
class PrecisionEnvelopeReport:
    """Conservative fixed-point output-envelope report for deployment checks."""

    operation: str
    output_codes: np.ndarray[Any, Any]
    overflow_mask: np.ndarray[Any, Any]
    abs_bound_codes: np.ndarray[Any, Any]
    output_fmt: QFormat
    underflow_mask: np.ndarray[Any, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operation, str) or not self.operation:
            raise ValueError("operation must be a non-empty string")
        if not isinstance(self.output_fmt, QFormat):
            raise TypeError("output_fmt must be a QFormat")

        raw_codes = np.asarray(self.output_codes)
        raw_bounds = np.asarray(self.abs_bound_codes)
        if not np.issubdtype(raw_codes.dtype, np.integer):
            raise TypeError("output_codes must contain integer fixed-point codes")
        if not np.issubdtype(raw_bounds.dtype, np.integer):
            raise TypeError("abs_bound_codes must contain integer fixed-point codes")

        codes = raw_codes.astype(np.int64, copy=True)
        bounds = raw_bounds.astype(np.int64, copy=True)
        mask = np.asarray(self.overflow_mask, dtype=bool)
        if self.underflow_mask is None:
            underflow = np.zeros(mask.shape, dtype=bool)
        else:
            underflow = np.asarray(self.underflow_mask, dtype=bool)

        if codes.ndim != 1:
            raise ValueError("output_codes must be a 1-D vector")
        if mask.ndim != 1:
            raise ValueError("overflow_mask must be a 1-D vector")
        if underflow.ndim != 1:
            raise ValueError("underflow_mask must be a 1-D vector")
        if bounds.ndim != 1:
            raise ValueError("abs_bound_codes must be a 1-D vector")
        if codes.shape != mask.shape or codes.shape != underflow.shape or codes.shape != bounds.shape:
            raise ValueError(
                "output_codes, overflow_mask, underflow_mask, and abs_bound_codes must have identical shape"
            )
        if np.any(bounds < 0):
            raise ValueError("abs_bound_codes must be non-negative")

        min_code, max_code = _fixed_integer_bounds(self.output_fmt)
        if np.any(codes < min_code) or np.any(codes > max_code):
            raise ValueError("output_codes exceed the configured output format")

        object.__setattr__(self, "output_codes", codes)
        object.__setattr__(self, "overflow_mask", mask.astype(bool, copy=True))
        object.__setattr__(self, "underflow_mask", underflow.astype(bool, copy=True))
        object.__setattr__(self, "abs_bound_codes", bounds)

    @property
    def output_count(self) -> int:
        """Number of output channels covered by this report."""
        return int(self.output_codes.size)

    @property
    def overflow_count(self) -> int:
        """Number of outputs that saturated during the producing operation."""
        return int(np.count_nonzero(self.overflow_mask))

    @property
    def observed_overflow_free(self) -> bool:
        """Whether the realised workload avoided fixed-point saturation."""
        return self.overflow_count == 0

    @property
    def underflow_count(self) -> int:
        """Number of nonzero outputs that collapsed below one output LSB."""
        return int(np.count_nonzero(self.underflow_mask))

    @property
    def observed_underflow_free(self) -> bool:
        """Whether the realised workload avoided sub-LSB output collapse."""
        return self.underflow_count == 0

    @property
    def conservative_safe_bound_code(self) -> int:
        """Largest symmetric absolute code accepted as overflow-free."""
        return (1 << (self.output_fmt.total_bits - 1)) - 1

    @property
    def max_abs_output_code(self) -> int:
        """Maximum absolute saturated output code observed in the workload."""
        if self.output_codes.size == 0:
            return 0
        return int(np.max(np.abs(self.output_codes.astype(object))))

    @property
    def max_abs_bound_code(self) -> int:
        """Maximum conservative absolute output bound for the workload."""
        if self.abs_bound_codes.size == 0:
            return 0
        return int(np.max(self.abs_bound_codes))

    @property
    def min_headroom_code(self) -> int:
        """Smallest conservative headroom, in fixed-point integer codes."""
        return self.conservative_safe_bound_code - self.max_abs_bound_code

    @property
    def conservative_overflow_free(self) -> bool:
        """Whether the absolute envelope proves the workload is in range."""
        return self.min_headroom_code >= 0

    def manifest(self) -> dict[str, bool | int | str]:
        """Deterministic envelope metadata for predeployment gates."""
        return {
            "operation": self.operation,
            "output_format": self.output_fmt.q_label,
            "output_count": self.output_count,
            "overflow_count": self.overflow_count,
            "underflow_count": self.underflow_count,
            "observed_overflow_free": self.observed_overflow_free,
            "observed_underflow_free": self.observed_underflow_free,
            "conservative_overflow_free": self.conservative_overflow_free,
            "max_abs_output_code": self.max_abs_output_code,
            "max_abs_bound_code": self.max_abs_bound_code,
            "conservative_safe_bound_code": self.conservative_safe_bound_code,
            "min_headroom_code": self.min_headroom_code,
        }


@dataclass(frozen=True)
class CompiledMixedDense:
    """Bit-true mixed fixed-point dense operator compiled from float weights."""

    quantized_weights: np.ndarray[Any, Any]
    tensor_scale: float
    fmt: QFormatMixed

    def __post_init__(self) -> None:
        if not isinstance(self.fmt, QFormatMixed):
            raise TypeError("fmt must be a QFormatMixed")
        if not math.isfinite(self.tensor_scale) or self.tensor_scale <= 0.0:
            raise ValueError("tensor_scale must be finite and positive")

        q_weights = np.asarray(self.quantized_weights, dtype=np.int64)
        if q_weights.ndim != 2:
            raise ValueError("quantized_weights must be a 2-D dense weight matrix")

        min_weight, max_weight = _fixed_integer_bounds(self.fmt.weight_fmt)
        if np.any(q_weights < min_weight) or np.any(q_weights > max_weight):
            raise ValueError("quantized_weights exceed the configured weight format")

        object.__setattr__(self, "quantized_weights", q_weights)

    @property
    def output_size(self) -> int:
        """Number of dense output channels."""
        return int(self.quantized_weights.shape[0])

    @property
    def input_size(self) -> int:
        """Number of dense input channels."""
        return int(self.quantized_weights.shape[1])

    @property
    def accumulator_divisor(self) -> float:
        """Raw-product divisor that converts Qw*Qa products into Qa codes."""
        return float(self.fmt.weight_fmt.scale) * self.tensor_scale

    def manifest(self) -> dict[str, bool | float | int | list[int] | str]:
        """Deterministic deployment metadata for host, Rust, and HDL emitters."""
        return {
            "operation": "dense_mixed_qformat",
            "input_size": self.input_size,
            "output_size": self.output_size,
            "weight_shape": [self.output_size, self.input_size],
            "tensor_scale": float(self.tensor_scale),
            "accumulator_divisor": self.accumulator_divisor,
            **self.fmt.metadata,
        }

    def _input_codes(self, inputs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        values = _finite_float_array(inputs, label="inputs")
        if values.ndim != 1:
            raise ValueError("inputs must be a 1-D vector")
        if values.shape[0] != self.input_size:
            raise ValueError(
                f"input length mismatch: expected {self.input_size}, got {values.shape[0]}"
            )
        return _quantize_fixed_array(
            values,
            self.fmt.accum_fmt,
            rounding=self.fmt.rounding,
            clip=True,
        ).astype(np.int64)

    def _raw_accumulator_products(self, input_codes: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        return self.quantized_weights.astype(np.int64) @ input_codes.astype(np.int64)

    def _forward_anomaly_masks(
        self, inputs: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        input_codes = self._input_codes(inputs)
        raw_products = self._raw_accumulator_products(input_codes)

        divisor = self.accumulator_divisor
        if divisor == self.fmt.weight_fmt.scale:
            accumulator_codes = np.floor_divide(raw_products, self.fmt.weight_fmt.scale)
        else:
            accumulator_codes = _round_scaled(
                raw_products.astype(np.float64) / divisor, self.fmt.rounding
            )

        min_accum, max_accum = _fixed_integer_bounds(self.fmt.accum_fmt)
        overflow = (accumulator_codes < min_accum) | (accumulator_codes > max_accum)
        underflow = (raw_products != 0) & (accumulator_codes == 0)
        clipped = np.clip(accumulator_codes, min_accum, max_accum).astype(np.int64)
        return clipped, overflow.astype(bool), (underflow & ~overflow).astype(bool)

    def forward_with_overflow(
        self, inputs: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Return saturated Q-accumulator codes and per-output overflow flags."""
        codes, overflow, _ = self._forward_anomaly_masks(inputs)
        return codes, overflow

    def forward_accumulator_codes(self, inputs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return saturated accumulator-format integer codes for dense outputs."""
        codes, _ = self.forward_with_overflow(inputs)
        return codes

    def precision_trap_report(self, inputs: np.ndarray[Any, Any]) -> PrecisionTrapReport:
        """Return saturation telemetry suitable for a hardware trap register."""
        codes, overflow, underflow = self._forward_anomaly_masks(inputs)
        return PrecisionTrapReport(
            operation="dense_mixed_qformat",
            output_codes=codes,
            overflow_mask=overflow,
            output_fmt=self.fmt.accum_fmt,
            underflow_mask=underflow,
        )

    def precision_envelope_report(self, inputs: np.ndarray[Any, Any]) -> PrecisionEnvelopeReport:
        """Return a conservative absolute-output envelope for this workload."""
        codes, overflow, underflow = self._forward_anomaly_masks(inputs)
        input_codes = self._input_codes(inputs)
        abs_products = np.abs(self.quantized_weights.astype(np.int64)) @ np.abs(
            input_codes.astype(np.int64)
        )

        divisor = self.accumulator_divisor
        if divisor == self.fmt.weight_fmt.scale:
            scale = self.fmt.weight_fmt.scale
            bounds = (abs_products + scale - 1) // scale
        else:
            bounds = np.ceil(abs_products.astype(np.float64) / divisor).astype(np.int64)

        return PrecisionEnvelopeReport(
            operation="dense_mixed_qformat",
            output_codes=codes,
            overflow_mask=overflow,
            abs_bound_codes=bounds.astype(np.int64),
            output_fmt=self.fmt.accum_fmt,
            underflow_mask=underflow,
        )

    def forward_float(self, inputs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return dense outputs reconstructed from saturated accumulator codes."""
        return self.forward_accumulator_codes(inputs).astype(np.float64) / self.fmt.accum_fmt.scale


def compile_dense_mixed_precision(
    weights: np.ndarray[Any, Any],
    fmt: QFormatMixed | None = None,
    *,
    clip: bool = True,
) -> CompiledMixedDense:
    """Compile a dense weight matrix into the mixed Q8.8/Q16.16 MAC contract."""
    mixed_fmt = fmt or QFormatMixed()
    if not isinstance(mixed_fmt, QFormatMixed):
        raise TypeError("fmt must be a QFormatMixed")

    weight_matrix = _finite_float_array(weights, label="weights")
    if weight_matrix.ndim != 2:
        raise ValueError("weights must be a 2-D dense matrix")

    quantized = quantize_weights(weight_matrix, fmt=mixed_fmt, clip=clip)
    if not isinstance(quantized, tuple):
        raise TypeError("mixed dense compilation requires QFormatMixed quantization")
    q_weights, tensor_scale = quantized
    return CompiledMixedDense(
        quantized_weights=np.asarray(q_weights, dtype=np.int64),
        tensor_scale=float(tensor_scale),
        fmt=mixed_fmt,
    )


@dataclass(frozen=True)
class CompiledBlockFloatingDense:
    """Dense operator compiled with shared-exponent block-floating weights."""

    mantissas: np.ndarray[Any, Any]
    exponents: np.ndarray[Any, Any]
    mode: BlockFloatingMode
    input_fmt: QFormat = Q16_16
    _weight_values: np.ndarray[Any, Any] = field(init=False, repr=False)
    _block_exponent_layout: BlockExponentLayout = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.mode, BlockFloatingMode):
            raise TypeError("mode must be a BlockFloatingMode")
        if not isinstance(self.input_fmt, QFormat):
            raise TypeError("input_fmt must be a QFormat")

        mantissas = np.asarray(self.mantissas, dtype=np.int64)
        exponents = np.asarray(self.exponents, dtype=np.int64).reshape(-1)
        if mantissas.ndim != 2:
            raise ValueError("mantissas must be a 2-D dense weight matrix")

        layout = self.mode.block_exponent_layout(int(mantissas.size))
        exponents = layout.validate_exponents(exponents, exponent_bits=self.mode.exponent_bits)

        if np.any(np.abs(mantissas) > self.mode.mantissa_range):
            raise ValueError("mantissas exceed the configured block-floating range")

        object.__setattr__(self, "mantissas", mantissas)
        object.__setattr__(self, "exponents", exponents)
        object.__setattr__(self, "_block_exponent_layout", layout)
        object.__setattr__(
            self, "_weight_values", self._reconstruct_weight_values(mantissas, exponents)
        )

    @property
    def output_size(self) -> int:
        """Number of dense output channels."""
        return int(self.mantissas.shape[0])

    @property
    def input_size(self) -> int:
        """Number of dense input channels."""
        return int(self.mantissas.shape[1])

    def _reconstruct_weight_values(
        self,
        mantissas: np.ndarray[Any, Any],
        exponents: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        if mantissas.size == 0:
            return mantissas.astype(np.float64)
        block_indices = np.arange(mantissas.size, dtype=np.int64) // self.mode.block_size
        unbiased = exponents[block_indices] - self.mode.exponent_bias
        scales = np.power(2.0, unbiased.astype(np.float64)).reshape(mantissas.shape)
        return mantissas.astype(np.float64) * scales

    @property
    def reconstructed_weights(self) -> np.ndarray[Any, Any]:
        """Float reconstruction of the compiled block-floating weight matrix."""
        return np.asarray(self._weight_values, dtype=np.float64).copy()

    def manifest(self) -> dict[str, Any]:
        """Deterministic deployment metadata for block-floating dense weights."""
        return {
            "operation": "dense_block_floating",
            "input_size": self.input_size,
            "output_size": self.output_size,
            "weight_shape": [self.output_size, self.input_size],
            "parameter_count": int(self.mantissas.size),
            "mantissa_bits": self.mode.mantissa_bits,
            "exponent_bits": self.mode.exponent_bits,
            "block_size": self.mode.block_size,
            "exponent_bias": self.mode.exponent_bias,
            "exponent_code_range": [0, (1 << self.mode.exponent_bits) - 1],
            "block_exponent_count": self._block_exponent_layout.exponent_count,
            "block_exponent_layout": self._block_exponent_layout.manifest(),
            "input_format": self.input_fmt.q_label,
        }

    def _input_values(self, inputs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        values = _finite_float_array(inputs, label="inputs")
        if values.ndim != 1:
            raise ValueError("inputs must be a 1-D vector")
        if values.shape[0] != self.input_size:
            raise ValueError(
                f"input length mismatch: expected {self.input_size}, got {values.shape[0]}"
            )
        input_codes = _quantize_fixed_array(
            values,
            self.input_fmt,
            rounding="nearest",
            clip=True,
        )
        return input_codes.astype(np.float64) / self.input_fmt.scale

    def forward_float(self, inputs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return dense outputs from BFP weights and quantised fixed-point inputs."""
        return np.asarray(self._weight_values, dtype=np.float64) @ self._input_values(inputs)

    def _forward_anomaly_masks(
        self, inputs: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        outputs = self.forward_float(inputs)
        codes = np.rint(outputs * self.input_fmt.scale).astype(np.int64)
        min_accum, max_accum = _fixed_integer_bounds(self.input_fmt)
        overflow = (codes < min_accum) | (codes > max_accum)
        underflow = (outputs != 0.0) & (codes == 0)
        clipped = np.clip(codes, min_accum, max_accum).astype(np.int64)
        return clipped, overflow.astype(bool), (underflow & ~overflow).astype(bool)

    def forward_with_overflow(
        self, inputs: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Return saturated fixed-point output codes and per-output overflow flags."""
        codes, overflow, _ = self._forward_anomaly_masks(inputs)
        return codes, overflow

    def forward_accumulator_codes(self, inputs: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return saturated output codes in the configured fixed-point input format."""
        codes, _ = self.forward_with_overflow(inputs)
        return codes

    def precision_trap_report(self, inputs: np.ndarray[Any, Any]) -> PrecisionTrapReport:
        """Return saturation telemetry suitable for a hardware trap register."""
        codes, overflow, underflow = self._forward_anomaly_masks(inputs)
        return PrecisionTrapReport(
            operation="dense_block_floating",
            output_codes=codes,
            overflow_mask=overflow,
            output_fmt=self.input_fmt,
            underflow_mask=underflow,
        )

    def precision_envelope_report(self, inputs: np.ndarray[Any, Any]) -> PrecisionEnvelopeReport:
        """Return a conservative absolute-output envelope for this workload."""
        codes, overflow, underflow = self._forward_anomaly_masks(inputs)
        input_values = self._input_values(inputs)
        abs_bound_values = np.abs(np.asarray(self._weight_values, dtype=np.float64)) @ np.abs(
            input_values
        )
        abs_bound_codes = np.ceil(abs_bound_values * self.input_fmt.scale)
        abs_bound_codes = np.minimum(abs_bound_codes, np.iinfo(np.int64).max).astype(np.int64)
        return PrecisionEnvelopeReport(
            operation="dense_block_floating",
            output_codes=codes,
            overflow_mask=overflow,
            abs_bound_codes=abs_bound_codes,
            output_fmt=self.input_fmt,
            underflow_mask=underflow,
        )


def compile_dense_block_floating(
    weights: np.ndarray[Any, Any],
    fmt: str = "BFP16E3X32",
    *,
    block_size: int | None = None,
    input_fmt: str | QFormat = Q16_16,
    clip: bool = True,
) -> CompiledBlockFloatingDense:
    """Compile a dense matrix into block-floating weights with Q-format inputs."""
    mode = BlockFloatingMode.from_aliases(fmt)
    selected_block_size = mode.block_size if block_size is None else block_size
    if selected_block_size <= 0:
        raise ValueError("block_size must be positive")
    if selected_block_size != mode.block_size:
        mode = BlockFloatingMode(
            mantissa_bits=mode.mantissa_bits,
            exponent_bits=mode.exponent_bits,
            block_size=selected_block_size,
        )

    weight_matrix = _finite_float_array(weights, label="weights")
    if weight_matrix.ndim != 2:
        raise ValueError("weights must be a 2-D dense matrix")

    mantissas, exponents = quantize_block_floating(
        weight_matrix,
        fmt=f"BFP{mode.mantissa_bits}E{mode.exponent_bits}X{mode.block_size}",
        block_size=mode.block_size,
        clip=clip,
    )
    return CompiledBlockFloatingDense(
        mantissas=mantissas,
        exponents=exponents,
        mode=mode,
        input_fmt=_coerce_q_format(input_fmt),
    )


def _encode_bfp_block(
    values: np.ndarray[Any, Any], mode: BlockFloatingMode, *, clip: bool
) -> tuple[int, np.ndarray[Any, Any]]:
    abs_max = float(np.max(np.abs(values))) if len(values) else 0.0
    if abs_max == 0.0:
        exponent = mode.exponent_bias
    else:
        unbiased_exp = int(math.ceil(math.log2(abs_max / mode.mantissa_range)))
        exponent = max(0, min((1 << mode.exponent_bits) - 1, unbiased_exp + mode.exponent_bias))

    exp_unbiased = exponent - mode.exponent_bias
    scale = 2.0**exp_unbiased
    encoded = np.rint(values / scale).astype(np.int64)

    if clip:
        encoded = np.clip(encoded, -mode.mantissa_range, mode.mantissa_range)
    return exponent, encoded


def quantize_block_floating(
    weights: np.ndarray[Any, Any],
    fmt: str,
    *,
    block_size: int = 32,
    clip: bool = True,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Quantize float weights into shared-exponent block-floating blocks."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    mode = BlockFloatingMode.from_aliases(fmt)
    explicit_block = "X" in fmt.upper()
    if explicit_block and block_size != mode.block_size:
        raise ValueError(
            f"Block size conflict for {fmt!r}: explicit format block_size is {mode.block_size}, "
            f"but block_size argument is {block_size}"
        )
    if not explicit_block:
        mode = BlockFloatingMode(
            mantissa_bits=mode.mantissa_bits,
            exponent_bits=mode.exponent_bits,
            block_size=block_size,
        )

    flat = np.asarray(weights, dtype=np.float64).reshape(-1)

    exponents = []
    quantized = np.empty_like(flat, dtype=np.int64)
    num_blocks = int(math.ceil(len(flat) / mode.block_size)) if flat.size else 0
    for block_idx in range(num_blocks):
        start = block_idx * mode.block_size
        end = min((block_idx + 1) * mode.block_size, len(flat))
        exp, encoded = _encode_bfp_block(flat[start:end], mode, clip=clip)
        quantized[start:end] = encoded
        exponents.append(exp)

    quantized = quantized.reshape(np.asarray(weights).shape)
    return quantized, np.array(exponents, dtype=np.int64)


def dequantize_block_floating(
    quantized: np.ndarray[Any, Any],
    exponents: np.ndarray[Any, Any],
    fmt: str,
) -> np.ndarray[Any, Any]:
    """Reconstruct floats from block-floating mantissas and exponents."""
    mode = BlockFloatingMode.from_aliases(fmt)
    if mode.block_size <= 0:
        raise ValueError(f"Invalid block size {mode.block_size}")

    flat = np.asarray(quantized, dtype=np.float64).reshape(-1)
    exps = np.asarray(exponents, dtype=np.int64).reshape(-1)

    num_blocks = int(math.ceil(flat.size / mode.block_size)) if flat.size else 0
    if exps.size != num_blocks:
        raise ValueError(f"Exponent count mismatch: expected {num_blocks}, got {int(exps.size)}")

    restored = np.empty_like(flat, dtype=np.float64)
    for idx in range(num_blocks):
        start = idx * mode.block_size
        end = min((idx + 1) * mode.block_size, flat.size)
        scale = 2.0 ** (int(exps[idx]) - mode.exponent_bias)
        restored[start:end] = flat[start:end] * scale

    return restored.reshape(np.asarray(quantized).shape)


def q_weights_to_sc_probabilities(
    quantized: np.ndarray[Any, Any], fmt: str | QFormat = "Q8.8"
) -> np.ndarray[Any, Any]:
    """Convert fixed-point quantized weights to SC probabilities in [0, 1]."""
    q = _coerce_q_format(fmt)
    min_int, max_int = _fixed_integer_bounds(q)
    return (quantized.astype(np.float64) - min_int) / (max_int - min_int)


def quantization_error(
    weights: np.ndarray[Any, Any],
    fmt: str | QFormat = "Q8.8",
    rounding: str = "nearest",
) -> dict[str, float]:
    """Compute quantization error statistics."""
    w = _finite_float_array(weights, label="weights")
    quantized = quantize_weights(w, fmt=fmt, rounding=rounding)
    if isinstance(quantized, tuple):
        raise TypeError("quantization_error expects a fixed-point QFormat, not QFormatMixed")
    recovered = dequantize_weights(quantized, fmt=fmt)
    error = w - recovered
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    signal_power = float(np.mean(w**2))
    snr = 10 * np.log10(signal_power / max(rmse**2, 1e-30))
    return {
        "max_abs_error": float(np.max(np.abs(error))),
        "mean_abs_error": mae,
        "rmse": rmse,
        "snr_db": float(snr),
    }


def dequantize_weights(
    quantized: np.ndarray[Any, Any],
    fmt: str | QFormat | QFormatMixed = "Q8.8",
    scale: float = 1.0,
) -> np.ndarray[Any, Any]:
    """Convert quantized fixed-point weights back to float."""
    if isinstance(fmt, str) and fmt.upper().startswith("BFP"):
        raise ValueError("BFP formats require dequantize_block_floating().")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")

    q = fmt.weight_fmt if isinstance(fmt, QFormatMixed) else _coerce_q_format(fmt)
    return quantized.astype(np.float64) / (q.scale * scale)


def dequantize(
    quantized: np.ndarray[Any, Any],
    fmt: str | QFormat | QFormatMixed = "Q8.8",
    scale: float = 1.0,
) -> np.ndarray[Any, Any]:
    """Alias matching the mixed-precision public API."""
    return dequantize_weights(quantized, fmt=fmt, scale=scale)
