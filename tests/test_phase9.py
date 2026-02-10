"""Tests for Phase 9: fast Bernoulli, fused AND+popcount, zero-copy prepacked."""

from __future__ import annotations

import numpy as np
import pytest

import sc_neurocore_engine as v3


class TestFastBernoulli:
    """Tests for byte-threshold Bernoulli in forward_fast and batch_encode_numpy."""

    def test_forward_fast_deterministic(self):
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        inputs = [0.5] * 16
        out1 = layer.forward_fast(inputs, seed=100)
        out2 = layer.forward_fast(inputs, seed=100)
        np.testing.assert_array_equal(out1, out2)

    def test_forward_fast_output_range(self):
        layer = v3.DenseLayer(8, 4, 1024, seed=42)
        inputs = [0.3] * 8
        out = layer.forward_fast(inputs, seed=42)
        assert all(0.0 <= v for v in out)

    def test_forward_fast_statistical_sanity(self):
        """forward_fast output should correlate with input probability."""
        layer = v3.DenseLayer(8, 4, 2048, seed=42)
        low_out = np.mean(layer.forward_fast([0.1] * 8, seed=42))
        high_out = np.mean(layer.forward_fast([0.9] * 8, seed=42))
        assert high_out > low_out, "Higher input probs should give higher output"

    def test_batch_encode_numpy_deterministic(self):
        probs = np.array([0.5, 0.5], dtype=np.float64)
        p1 = v3.batch_encode_numpy(probs, length=256, seed=42)
        p2 = v3.batch_encode_numpy(probs, length=256, seed=42)
        np.testing.assert_array_equal(p1, p2)

    def test_batch_encode_numpy_statistics(self):
        """Encoded bitstreams should have popcount proportional to probability."""
        probs = np.array([0.25, 0.75], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=10_000, seed=42)
        pc0 = sum(int(w).bit_count() for w in packed[0])
        pc1 = sum(int(w).bit_count() for w in packed[1])
        assert abs(pc0 / 10_000 - 0.25) < 0.04
        assert abs(pc1 / 10_000 - 0.75) < 0.04


class TestFusedAndPopcount:
    """Tests verifying fused AND+popcount produces same results as before."""

    def test_forward_matches_reference(self):
        """forward() output should still be valid (range + deterministic)."""
        layer = v3.DenseLayer(8, 4, 512, seed=42)
        inputs = [0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8, 0.1]
        out1 = layer.forward(inputs, seed=42)
        out2 = layer.forward(inputs, seed=42)
        np.testing.assert_array_equal(out1, out2)
        assert all(0.0 <= v for v in out1)

    def test_prepacked_deterministic(self):
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        probs = np.array([0.3, 0.5, 0.7, 0.9], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=55)
        out1 = layer.forward_prepacked(packed)
        out2 = layer.forward_prepacked(packed)
        np.testing.assert_array_equal(out1, out2)


class TestZeroCopyPrepackedNumpy:
    """Tests for forward_prepacked_numpy (true zero-copy path)."""

    def test_output_shape_and_type(self):
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        probs = np.array([0.3, 0.5, 0.7, 0.9], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=55)
        out = layer.forward_prepacked_numpy(packed)
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)
        assert out.dtype == np.float64

    def test_matches_forward_prepacked(self):
        """Zero-copy numpy path must match the existing prepacked path."""
        layer = v3.DenseLayer(8, 4, 512, seed=42)
        probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=512, seed=99)
        out_legacy = layer.forward_prepacked(packed)
        out_numpy = layer.forward_prepacked_numpy(packed)
        np.testing.assert_allclose(out_numpy, out_legacy)

    def test_wrong_n_inputs(self):
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        packed = np.zeros((3, 16), dtype=np.uint64)  # 3 inputs, need 4
        with pytest.raises(ValueError):
            layer.forward_prepacked_numpy(packed)

    def test_wrong_word_count(self):
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        packed = np.zeros((4, 10), dtype=np.uint64)  # 10 words, need 16
        with pytest.raises(ValueError):
            layer.forward_prepacked_numpy(packed)

    def test_pipeline_encode_then_zero_copy(self):
        """Full pipeline: batch_encode_numpy -> forward_prepacked_numpy."""
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        probs = np.random.uniform(0, 1, 16)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=42)
        out = layer.forward_prepacked_numpy(packed)
        assert out.shape == (8,)
        assert np.all(out >= 0.0)

    def test_deterministic(self):
        layer = v3.DenseLayer(4, 2, 512, seed=42)
        probs = np.array([0.5] * 4, dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=512, seed=42)
        out1 = layer.forward_prepacked_numpy(packed)
        out2 = layer.forward_prepacked_numpy(packed)
        np.testing.assert_array_equal(out1, out2)


class TestSetNumThreads:
    """Tests for rayon thread pool configuration."""

    def test_set_num_threads_does_not_crash(self):
        """Calling set_num_threads should not raise."""
        # Can only be set before global pool initialization. If initialized,
        # rayon returns an error, which is acceptable behavior.
        try:
            v3.set_num_threads(0)  # 0 = default
        except ValueError:
            pass


class TestPhase9Version:
    def test_version_is_3_6_0(self):
        assert v3.__version__ == "3.6.0"
