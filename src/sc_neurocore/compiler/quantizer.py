# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Weight quantization for hardware deployment

"""Quantize trained float weights to Q-format fixed-point for SC hardware.

from sc_neurocore.compiler.quantizer import quantize_weights

# After training, quantize for FPGA deployment
q_weights = quantize_weights(float_weights, format="Q8.8")
sc_probs = q_weights_to_sc_probabilities(q_weights, format="Q8.8")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
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
        Q-format string, e.g. "Q8.8" (8 integer + 8 fractional = 16-bit signed).
    rounding : str
        "nearest" (round half to even), "stochastic" (probabilistic rounding),
        or "floor" (truncate toward negative infinity).
    clip : bool
        If True, clip values to the representable range before quantization.

    Returns
    -------
    np.ndarray
        Integer array (same shape) in the Q-format representation.
        To recover the float: result / (2^fraction_bits).
    """
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


def dequantize_weights(quantized: np.ndarray[Any, Any], fmt: str = "Q8.8") -> np.ndarray[Any, Any]:
    """Convert quantized integer weights back to float."""
    q = QFormat.from_string(fmt)
    return quantized.astype(np.float64) / q.scale


def q_weights_to_sc_probabilities(quantized: np.ndarray[Any, Any], fmt: str = "Q8.8") -> np.ndarray[Any, Any]:
    """Convert quantized weights to SC probabilities in [0, 1].

    Maps the Q-format range [min, max] linearly to [0, 1] for
    unipolar SC encoding.
    """
    q = QFormat.from_string(fmt)
    min_int = -(1 << (q.total_bits - 1))
    max_int = (1 << (q.total_bits - 1)) - 1
    return (quantized.astype(np.float64) - min_int) / (max_int - min_int)


def quantization_error(weights: np.ndarray[Any, Any], fmt: str = "Q8.8", rounding: str = "nearest") -> dict[str, float]:
    """Compute quantization error statistics.

    Returns
    -------
    dict with keys: max_abs_error, mean_abs_error, rmse, snr_db
    """
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
