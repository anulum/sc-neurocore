# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mixed-precision dense quantization

"""Mixed-precision dense operator compilation and utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .fixed_point_quantization import (
    _finite_float_array,
    _quantize_fixed_array,
    _round_scaled,
    quantize_weights,
)
from .q_format import QFormatMixed
from .quantization_reports import (
    PrecisionEnvelopeReport,
    PrecisionTrapReport,
    _fixed_integer_bounds,
)


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
        products: np.ndarray[Any, Any] = self.quantized_weights.astype(
            np.int64
        ) @ input_codes.astype(np.int64)
        return products

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
