# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Block floating-point quantization

"""Block floating-point quantization implementation and utilities."""

from __future__ import annotations

import math
from dataclasses import field, dataclass
from typing import Any

import numpy as np

from .q_format import QFormat, Q16_16
from .block_floating import BlockFloatingMode, BlockExponentLayout
from .quantization_reports import (
    PrecisionTrapReport,
    PrecisionEnvelopeReport,
    _fixed_integer_bounds,
)
from .fixed_point_quantization import (
    _finite_float_array,
    _quantize_fixed_array,
    _coerce_q_format,
)


@dataclass(frozen=True)
class CompiledBlockFloatingDense:
    """Dense operator compiled with shared-exponent block-floating weights."""

    mantissas: np.ndarray[Any, Any]
    exponents: np.ndarray[Any, Any]
    mode: BlockFloatingMode
    input_fmt: QFormat = Q16_16
    _weight_values: np.ndarray[Any, Any] = field(init=False, repr=False)
    # _block_exponent_layout is intentionally private and used for internal validation
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
