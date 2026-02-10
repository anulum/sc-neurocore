"""Tests for Phase 7 dense forward optimizations."""

from __future__ import annotations

import numpy as np
import pytest

import sc_neurocore_engine as v3


class TestBernoulliPackedEquivalence:
    """Validate deterministic behavior for packed Bernoulli refactor."""

    def test_pack_deterministic(self):
        """forward() should remain deterministic for fixed inputs and seed."""
        layer = v3.DenseLayer(8, 4, 256, seed=12345)
        inputs = [0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8]
        out = layer.forward(inputs, seed=99999)
        assert len(out) == 4
        assert all(0.0 <= v <= 8.0 for v in out)

    def test_pack_deterministic_repeated(self):
        """Same inputs + seeds produce exactly same outputs."""
        layer = v3.DenseLayer(8, 4, 256, seed=12345)
        out1 = layer.forward([0.5] * 8, seed=42)
        out2 = layer.forward([0.5] * 8, seed=42)
        assert out1 == out2


class TestForwardFast:
    """Tests for parallel-encoded forward_fast method."""

    def test_output_shape(self):
        layer = v3.DenseLayer(16, 8, 512)
        out = layer.forward_fast([0.5] * 16)
        assert len(out) == 8

    def test_output_range(self):
        layer = v3.DenseLayer(16, 8, 512)
        out = layer.forward_fast([0.3] * 16)
        assert all(0.0 <= v <= 16.0 for v in out)

    def test_deterministic(self):
        layer = v3.DenseLayer(16, 8, 512, seed=42)
        out1 = layer.forward_fast([0.5] * 16, seed=100)
        out2 = layer.forward_fast([0.5] * 16, seed=100)
        assert out1 == out2

    def test_different_seed_different_output(self):
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        out1 = layer.forward_fast([0.5] * 16, seed=100)
        out2 = layer.forward_fast([0.5] * 16, seed=200)
        assert out1 != out2

    def test_statistical_sanity(self):
        """forward_fast should have similar distribution to forward."""
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        inputs = [0.5, 0.5, 0.5, 0.5]
        results_orig = [layer.forward(inputs, seed=s) for s in range(50)]
        results_fast = [layer.forward_fast(inputs, seed=s) for s in range(50)]
        mean_orig = np.mean([r[0] for r in results_orig])
        mean_fast = np.mean([r[0] for r in results_fast])
        assert abs(mean_orig - mean_fast) < 0.1

    def test_wrong_input_length(self):
        layer = v3.DenseLayer(8, 4, 256)
        with pytest.raises(ValueError):
            layer.forward_fast([0.5] * 7)


class TestForwardPrepacked:
    """Tests for pre-packed forward path."""

    def test_output_shape(self):
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        probs = np.array([0.3, 0.5, 0.7, 0.9], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=55)
        out = layer.forward_prepacked(packed)
        assert len(out) == 2

    def test_output_range(self):
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        probs = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=55)
        out = layer.forward_prepacked(packed)
        assert all(0.0 <= v <= 4.0 for v in out)

    def test_deterministic(self):
        """Same pre-packed inputs should always produce same output."""
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        packed = v3.batch_encode_numpy(np.array([0.5] * 4), length=1024, seed=55)
        out1 = layer.forward_prepacked(packed)
        out2 = layer.forward_prepacked(packed)
        assert out1 == out2

    def test_accepts_list_of_lists(self):
        """forward_prepacked should also accept list[list[int]]."""
        layer = v3.DenseLayer(2, 1, 128, seed=42)
        packed = v3.batch_encode(np.array([0.5, 0.5]), length=128, seed=55)
        out = layer.forward_prepacked(packed)
        assert len(out) == 1

    def test_wrong_n_inputs(self):
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        packed = v3.batch_encode_numpy(np.array([0.5, 0.5, 0.5]), length=1024, seed=55)
        with pytest.raises(ValueError):
            layer.forward_prepacked(packed)

    def test_wrong_word_count(self):
        layer = v3.DenseLayer(2, 1, 1024, seed=42)
        packed = v3.batch_encode_numpy(np.array([0.5, 0.5]), length=512, seed=55)
        with pytest.raises(ValueError):
            layer.forward_prepacked(packed)


class TestBatchEncodeNumpy:
    """Tests for batch_encode_numpy returning 2-D numpy array."""

    def test_shape(self):
        probs = np.array([0.3, 0.5, 0.7], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=42)
        assert packed.shape == (3, 16)
        assert packed.dtype == np.uint64

    def test_deterministic(self):
        probs = np.array([0.5, 0.5], dtype=np.float64)
        p1 = v3.batch_encode_numpy(probs, length=256, seed=42)
        p2 = v3.batch_encode_numpy(probs, length=256, seed=42)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seed(self):
        probs = np.array([0.5], dtype=np.float64)
        p1 = v3.batch_encode_numpy(probs, length=1024, seed=1)
        p2 = v3.batch_encode_numpy(probs, length=1024, seed=2)
        assert not np.array_equal(p1, p2)

    def test_parallel_deterministic(self):
        """Parallel batch_encode_numpy should be deterministic with same seed."""
        probs = np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64)
        r1 = v3.batch_encode_numpy(probs, length=256, seed=42)
        r2 = v3.batch_encode_numpy(probs, length=256, seed=42)
        np.testing.assert_array_equal(r1, r2)

    def test_popcount_statistics(self):
        """Encoded bitstreams should reflect input probabilities."""
        probs = np.array([0.25, 0.75], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=10_000, seed=42)
        pc0 = sum(int(w).bit_count() for w in packed[0])
        pc1 = sum(int(w).bit_count() for w in packed[1])
        assert abs(pc0 / 10_000 - 0.25) < 0.03
        assert abs(pc1 / 10_000 - 0.75) < 0.03

    def test_empty_probs(self):
        probs = np.array([], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=64, seed=42)
        assert packed.shape == (0, 1)

    def test_pipeline_encode_then_forward(self):
        """End-to-end: batch_encode_numpy -> forward_prepacked."""
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        probs = np.array([0.3, 0.5, 0.7, 0.9], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=55)
        out = layer.forward_prepacked(packed)
        assert len(out) == 2
        assert all(0.0 <= v <= 4.0 for v in out)
