# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fixed-point quantization

"""Fixed-point quantization implementation and utilities."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .q_format import QFormat, QFormatMixed


def _fixed_integer_bounds(q: QFormat) -> tuple[int, int]:
    return -(1 << (q.total_bits - 1)), (1 << (q.total_bits - 1)) - 1


def _coerce_q_format(fmt: str | QFormat) -> QFormat:
    if isinstance(fmt, QFormat):
        return fmt
    if isinstance(fmt, str):
        return QFormat.from_string(fmt)
    raise TypeError(f"Expected QFormat or Q-format string, got {type(fmt)!r}")


def _finite_float_array(values: np.ndarray[Any, Any], *, label: str) -> np.ndarray[Any, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must contain only finite values")
    return arr


def _round_scaled(scaled: np.ndarray[Any, Any], rounding: str) -> np.ndarray[Any, Any]:
    if rounding == "nearest":
        rounded: np.ndarray[Any, Any] = np.rint(scaled).astype(np.int64)
        return rounded
    if rounding == "stochastic":
        floor = np.floor(scaled)
        probability = scaled - floor
        stochastic: np.ndarray[Any, Any] = (
            floor + (np.random.random(scaled.shape) < probability)
        ).astype(np.int64)
        return stochastic
    if rounding == "floor":
        floored: np.ndarray[Any, Any] = np.floor(scaled).astype(np.int64)
        return floored
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
