# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for block-floating dense compilation and quantisation

"""Contracts for block-floating dense compilation, quantisation and validation."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from sc_neurocore.compiler.block_floating_quantization import (
    compile_dense_block_floating,
    dequantize_block_floating,
    quantize_block_floating,
)

_W = np.array([[0.5, -0.25, 0.125, 0.0], [1.0, -1.0, 0.5, 0.25]], dtype=np.float64)
_INPUTS = np.array([0.1, 0.2, 0.3, 0.4])


def test_compile_reconstructs_and_runs_forward_paths() -> None:
    """A compiled BFP dense layer reconstructs weights and runs the forward paths."""
    compiled = compile_dense_block_floating(_W, fmt="BFP16E3X32")

    assert compiled.reconstructed_weights.shape == _W.shape
    assert compiled.forward_float(_INPUTS).shape == (2,)
    assert compiled.forward_accumulator_codes(_INPUTS).shape == (2,)


def test_compile_rejects_non_positive_block_size() -> None:
    """compile rejects a non-positive explicit block size."""
    with pytest.raises(ValueError, match="block_size must be positive"):
        compile_dense_block_floating(_W, block_size=0)


def test_compile_rebuilds_mode_for_custom_block_size() -> None:
    """compile rebuilds the block-floating mode when a different block size is requested."""
    compiled = compile_dense_block_floating(_W, fmt="BFP16E3X32", block_size=64)
    assert compiled.mode.block_size == 64


def test_forward_rejects_non_1d_inputs() -> None:
    """forward rejects a non-1-D input vector."""
    compiled = compile_dense_block_floating(_W)
    with pytest.raises(ValueError, match="1-D vector"):
        compiled.forward_float(np.zeros((2, 4)))


def test_compile_handles_empty_weight_matrix() -> None:
    """Compiling an empty weight matrix yields an empty reconstruction."""
    compiled = compile_dense_block_floating(np.zeros((0, 4)))
    assert compiled.reconstructed_weights.size == 0


def test_quantize_all_zero_block_uses_bias_exponent() -> None:
    """A zero-magnitude block encodes to all-zero mantissas at the bias exponent."""
    mantissas, _ = quantize_block_floating(np.zeros((2, 4)), fmt="BFP16E3X32")
    assert mantissas.shape == (2, 4)
    assert np.all(mantissas == 0)


def test_quantize_rejects_non_positive_block_size() -> None:
    """quantize rejects a non-positive block size."""
    with pytest.raises(ValueError, match="block_size must be positive"):
        quantize_block_floating(_W, fmt="BFP16E3", block_size=0)


def test_quantize_rebuilds_mode_without_explicit_block() -> None:
    """quantize rebuilds the mode from the block_size argument when fmt omits it."""
    mantissas, _ = quantize_block_floating(_W, fmt="BFP16E3", block_size=4)
    assert mantissas.shape == _W.shape


def test_dequantize_rejects_exponent_count_mismatch() -> None:
    """dequantize rejects a wrong number of exponents."""
    mantissas, _ = quantize_block_floating(_W, fmt="BFP16E3X32")
    with pytest.raises(ValueError, match="Exponent count mismatch"):
        dequantize_block_floating(mantissas, np.array([1, 2, 3], dtype=np.int64), fmt="BFP16E3X32")


@pytest.mark.parametrize(
    "override",
    [
        {"mode": "not a mode"},
        {"input_fmt": "not a qformat"},
        {"mantissas": np.zeros(4)},
        {"mantissas": np.full((2, 4), 10**9, dtype=np.int64)},
    ],
)
def test_compiled_dense_rejects_invalid_fields(override: dict[str, Any]) -> None:
    """CompiledBlockFloatingDense rejects a malformed mode, input format or mantissas."""
    compiled = compile_dense_block_floating(_W)
    with pytest.raises((TypeError, ValueError)):
        dataclasses.replace(compiled, **override)
