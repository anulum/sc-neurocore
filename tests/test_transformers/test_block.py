"""Tests for StochasticTransformerBlock forward behavior."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.transformers.block import StochasticTransformerBlock


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_block_init_shapes():
    """FFN layers should match expected input/output sizes."""
    block = StochasticTransformerBlock(d_model=4, n_heads=1, length=16)
    assert block.ffn_1.n_inputs == 4
    assert block.ffn_1.n_neurons == 16
    assert block.ffn_2.n_inputs == 16
    assert block.ffn_2.n_neurons == 4


def test_block_forward_shape_1d():
    """1D input should yield (d_model,) output."""
    block = StochasticTransformerBlock(d_model=4, n_heads=1, length=16)
    x = np.array([0.1, 0.2, 0.3, 0.4])
    out = block.forward(x)
    assert out.shape == (4,)


def test_block_forward_shape_2d_single_token():
    """2D single-token input should yield (d_model,) output."""
    block = StochasticTransformerBlock(d_model=3, n_heads=1, length=16)
    x = np.array([[0.1, 0.2, 0.3]])
    out = block.forward(x)
    assert out.shape == (3,)


def test_block_forward_multi_token_raises():
    """Multi-token inputs should error due to shape mismatch."""
    block = StochasticTransformerBlock(d_model=2, n_heads=1, length=16)
    x = np.array([[0.1, 0.2], [0.3, 0.4]])
    with pytest.raises(ValueError):
        _ = block.forward(x)


def test_block_output_finite():
    """Output values should be finite."""
    block = StochasticTransformerBlock(d_model=4, n_heads=1, length=16)
    x = np.random.random(4)
    out = block.forward(x)
    assert np.all(np.isfinite(out))


def test_block_length_propagation():
    """Configured length should propagate to FFN layers."""
    block = StochasticTransformerBlock(d_model=4, n_heads=1, length=64)
    assert block.ffn_1.length == 64
    assert block.ffn_2.length == 64


def test_block_deterministic_with_seed():
    """Numpy seed should make outputs repeatable for same input."""
    x = np.array([0.2, 0.4, 0.6, 0.8])

    # Seed before creating block and running forward
    np.random.seed(123)
    block_a = StochasticTransformerBlock(d_model=4, n_heads=1, length=16)
    out_a = block_a.forward(x)

    # Seed again to reset random state completely
    np.random.seed(123)
    block_b = StochasticTransformerBlock(d_model=4, n_heads=1, length=16)
    out_b = block_b.forward(x)

    assert np.allclose(out_a, out_b)


def test_block_output_not_nan():
    """Output should not contain NaNs."""
    block = StochasticTransformerBlock(d_model=3, n_heads=1, length=16)
    x = np.array([0.1, 0.2, 0.3])
    out = block.forward(x)
    assert not np.isnan(out).any()


def test_block_forward_accepts_float_input():
    """Float input arrays should be accepted."""
    block = StochasticTransformerBlock(d_model=2, n_heads=1, length=16)
    x = np.array([0.5, 0.25], dtype=float)
    out = block.forward(x)
    assert out.shape == (2,)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_block_perf_small():
    """Benchmark a small block forward pass."""
    block = StochasticTransformerBlock(d_model=8, n_heads=1, length=32)
    x = np.random.random(8)
    start = time.perf_counter()
    _ = block.forward(x)
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0
