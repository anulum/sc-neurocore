# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for StochasticTransformerBlock forward behavior

"""Tests for StochasticTransformerBlock forward behavior."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.transformers.block import StochasticTransformerBlock
from tests.performance_guard import assert_load_tolerant_throughput


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_block_init_shapes():
    """FFN layers should match expected input/output sizes."""
    block = StochasticTransformerBlock(d_model=4, n_heads=1, length=16)
    assert len(block.attention_heads) == 1
    assert block.attention_heads[0].dim_k == 4
    assert block.ffn_1.n_inputs == 4
    assert block.ffn_1.n_neurons == 16
    assert block.ffn_2.n_inputs == 16
    assert block.ffn_2.n_neurons == 4


def test_block_initialises_all_attention_heads():
    """Multi-head blocks should allocate one attention primitive per head."""
    block = StochasticTransformerBlock(d_model=8, n_heads=4, length=16)
    assert len(block.attention_heads) == 4
    assert [head.dim_k for head in block.attention_heads] == [2, 2, 2, 2]


def test_block_rejects_invalid_head_partition():
    """Each head must receive an equal non-empty feature subspace."""
    with pytest.raises(ValueError, match="d_model must be divisible by n_heads"):
        StochasticTransformerBlock(d_model=6, n_heads=4, length=16)


def test_block_forward_shape_1d():
    """1D input should yield (d_model,) output."""
    block = StochasticTransformerBlock(d_model=4, n_heads=1, length=16)
    x = np.array([0.1, 0.2, 0.3, 0.4])
    out = block.forward(x)
    assert out.shape == (4,)


def test_block_forward_shape_2d_single_token():
    """2D single-token input should yield (1, d_model) output."""
    block = StochasticTransformerBlock(d_model=3, n_heads=1, length=16)
    x = np.array([[0.1, 0.2, 0.3]])
    out = block.forward(x)
    assert out.shape == (1, 3)


def test_block_forward_multi_token():
    """Multi-token inputs should produce (seq_len, d_model) output."""
    block = StochasticTransformerBlock(d_model=2, n_heads=1, length=16)
    x = np.array([[0.1, 0.2], [0.3, 0.4]])
    out = block.forward(x)
    assert out.shape == (2, 2)
    assert np.all(np.isfinite(out))


def test_block_multi_head_uses_disjoint_feature_slices():
    """Each attention head must receive its own contiguous feature slice."""

    class RecordingHead:
        def __init__(self, value: float) -> None:
            self.value = value
            self.calls: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []

        def forward(self, Q, K, V):  # noqa: N803 - mirrors attention API
            q = np.asarray(Q)
            k = np.asarray(K)
            v = np.asarray(V)
            self.calls.append((q.shape, k.shape, v.shape))
            return np.full((q.shape[0], v.shape[1]), self.value, dtype=float)

    block = StochasticTransformerBlock(d_model=4, n_heads=2, length=16)
    block.attention_heads = [RecordingHead(0.25), RecordingHead(0.75)]  # type: ignore[list-item]

    out = block._multi_head_attention(np.array([[0.1, 0.2, 0.3, 0.4]]))

    assert out.shape == (1, 4)
    np.testing.assert_allclose(out, np.array([[0.25, 0.25, 0.75, 0.75]]))
    assert block.attention_heads[0].calls == [((1, 2), (1, 2), (1, 2))]  # type: ignore[attr-defined]
    assert block.attention_heads[1].calls == [((1, 2), (1, 2), (1, 2))]  # type: ignore[attr-defined]


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
    assert_load_tolerant_throughput(
        label="transformer block run",
        observed_per_second=1.0 / elapsed,
        strict_minimum_per_second=1.0 / 3.0,
    )
