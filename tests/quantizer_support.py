# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_quantizer.py

from __future__ import annotations

"""Tests for quantizer: float weights → Q-format fixed-point → SC probabilities."""
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sc_neurocore.compiler.static_analysis import prove_fixed_point_envelope
from sc_neurocore.compiler.quantizer import (
    QFormat,
    QFormatMixed,
    Q8_8,
    Q16_16,
    BlockFloatingMode,
    CompiledMixedDense,
    PrecisionEnvelopeReport,
    PrecisionTrapReport,
    compile_dense_block_floating,
    compile_dense_mixed_precision,
    parse_precision_format,
    quantize_block_floating,
    dequantize,
    dequantize_block_floating,
    quantize_weights,
    dequantize_weights,
    q_weights_to_sc_probabilities,
    quantization_error,
)

__all__ = ['np', 'pytest', 'given', 'settings', 'st', 'prove_fixed_point_envelope', 'QFormat', 'QFormatMixed', 'Q8_8', 'Q16_16', 'BlockFloatingMode', 'CompiledMixedDense', 'PrecisionEnvelopeReport', 'PrecisionTrapReport', 'compile_dense_block_floating', 'compile_dense_mixed_precision', 'parse_precision_format', 'quantize_block_floating', 'dequantize', 'dequantize_block_floating', 'quantize_weights', 'dequantize_weights', 'q_weights_to_sc_probabilities', 'quantization_error']
