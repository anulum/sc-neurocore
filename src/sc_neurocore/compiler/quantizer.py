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
from typing import Any

import numpy as np


@dataclass(frozen=True)
class QFormat:
    """Fixed-point Q-format specification."""

    integer_bits: int
    fraction_bits: int

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

    @classmethod
    def from_string(cls, fmt: str) -> QFormat:
        """Parse 'Q8.8', 'Q4.12', etc."""
        fmt = fmt.strip().upper()
        if not fmt.startswith("Q") or "." not in fmt:
            raise ValueError(f"Expected format like 'Q8.8', got {fmt!r}")
        parts = fmt[1:].split(".")
        return cls(integer_bits=int(parts[0]), fraction_bits=int(parts[1]))


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


def quantize_weights(
    weights: np.ndarray[Any, Any],
    fmt: str = "Q8.8",
    rounding: str = "nearest",
    clip: bool = True,
) -> np.ndarray[Any, Any]:
    """Quantize float weights to fixed-point integers.

    Parameters
    ----------
    weights : np.ndarray
        Float weight matrix (any shape).
    fmt : str
        Q-format string, e.g. ``"Q8.8"``.
    rounding : str
        ``nearest`` (round half to even), ``stochastic``, or ``floor``.
    clip : bool
        If True, clip values to the representable range before quantization.
    """
    if fmt.upper().startswith("BFP"):
        raise ValueError(
            "Block-floating formats are supported via quantize_block_floating(); "
            "quantize_weights() is fixed-point only."
        )

    q = QFormat.from_string(fmt)
    w = np.asarray(weights, dtype=np.float64)

    if clip:
        w = np.clip(w, q.min_val, q.max_val)

    scaled = w * q.scale

    if rounding == "nearest":
        quantized = np.rint(scaled).astype(np.int64)
    elif rounding == "stochastic":
        floor = np.floor(scaled)
        prob = scaled - floor
        quantized = (floor + (np.random.random(w.shape) < prob)).astype(np.int64)
    elif rounding == "floor":
        quantized = np.floor(scaled).astype(np.int64)
    else:
        raise ValueError(
            f"Unknown rounding mode: {rounding!r}. Use 'nearest', 'stochastic', or 'floor'."
        )

    min_int = -(1 << (q.total_bits - 1))
    max_int = (1 << (q.total_bits - 1)) - 1
    return np.clip(quantized, min_int, max_int)


def _encode_bfp_block(values: np.ndarray[Any, Any], mode: BlockFloatingMode, *, clip: bool) -> tuple[int, np.ndarray[Any, Any]]:
    abs_max = float(np.max(np.abs(values))) if len(values) else 0.0
    if abs_max == 0.0:
        exponent = mode.exponent_bias
    else:
        unbiased_exp = int(math.floor(math.log2(abs_max)))
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
    quantized: np.ndarray[Any, Any], fmt: str = "Q8.8"
) -> np.ndarray[Any, Any]:
    """Convert fixed-point quantized weights to SC probabilities in [0, 1]."""
    q = QFormat.from_string(fmt)
    min_int = -(1 << (q.total_bits - 1))
    max_int = (1 << (q.total_bits - 1)) - 1
    return (quantized.astype(np.float64) - min_int) / (max_int - min_int)


def quantization_error(
    weights: np.ndarray[Any, Any], fmt: str = "Q8.8", rounding: str = "nearest"
) -> dict[str, float]:
    """Compute quantization error statistics."""
    quantized = quantize_weights(weights, fmt=fmt, rounding=rounding)
    recovered = dequantize_weights(quantized, fmt=fmt)
    error = weights - recovered
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    signal_power = float(np.mean(weights**2))
    snr = 10 * np.log10(signal_power / max(rmse**2, 1e-30))
    return {
        "max_abs_error": float(np.max(np.abs(error))),
        "mean_abs_error": mae,
        "rmse": rmse,
        "snr_db": float(snr),
    }


def dequantize_weights(quantized: np.ndarray[Any, Any], fmt: str = "Q8.8") -> np.ndarray[Any, Any]:
    """Convert quantized fixed-point weights back to float."""
    if fmt.upper().startswith("BFP"):
        raise ValueError("BFP formats require dequantize_block_floating().")
    q = QFormat.from_string(fmt)
    return quantized.astype(np.float64) / q.scale
