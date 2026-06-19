# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantization compatibility facade

"""Compatibility module for the quantization core import surface."""

from __future__ import annotations

from .quantizer import (
    Q8_8,
    Q16_16,
    BlockExponentLayout,
    BlockFloatingMode,
    CompiledBlockFloatingDense,
    CompiledMixedDense,
    PrecisionEnvelopeReport,
    PrecisionTrapReport,
    QFormat,
    QFormatMixed,
    RoundingMode,
    compile_dense_block_floating,
    compile_dense_mixed_precision,
    dequantize,
    dequantize_block_floating,
    dequantize_weights,
    parse_precision_format,
    q_weights_to_sc_probabilities,
    quantization_error,
    quantize_block_floating,
    quantize_weights,
)

__all__ = [
    "BlockExponentLayout",
    "BlockFloatingMode",
    "CompiledBlockFloatingDense",
    "CompiledMixedDense",
    "PrecisionEnvelopeReport",
    "PrecisionTrapReport",
    "Q8_8",
    "Q16_16",
    "QFormat",
    "QFormatMixed",
    "RoundingMode",
    "compile_dense_block_floating",
    "compile_dense_mixed_precision",
    "dequantize",
    "dequantize_block_floating",
    "dequantize_weights",
    "parse_precision_format",
    "q_weights_to_sc_probabilities",
    "quantization_error",
    "quantize_block_floating",
    "quantize_weights",
]
