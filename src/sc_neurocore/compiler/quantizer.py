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
from dataclasses import dataclass
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
        return self.exponent_bias

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
        }

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
    raise ValueError(f"Unknown rounding mode: {rounding!r}. Use 'nearest', 'stochastic', or 'floor'.")


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


def _encode_bfp_block(values: np.ndarray[Any, Any], mode: BlockFloatingMode, *, clip: bool) -> tuple[int, np.ndarray[Any, Any]]:
    abs_max = float(np.max(np.abs(values))) if len(values) else 0.0
    if abs_max == 0.0:
        exponent = mode.exponent_bias
    else:
        unbiased_exp = int(math.ceil(math.log2(abs_max / mode.mantissa_range)))
        exponent = max(0, min((1 << mode.exponent_bits) - 1, unbiased_exp + mode.exponent_bias))

    exp_unbiased = exponent - mode.exponent_bias
    scale = 2.0 ** exp_unbiased
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
        raise ValueError(
            f"Exponent count mismatch: expected {num_blocks}, got {int(exps.size)}"
        )

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
