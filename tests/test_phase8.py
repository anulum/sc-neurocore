"""Tests for Phase 8: forward_numpy and parallel batch_encode_numpy."""

from __future__ import annotations

import numpy as np
import pytest

import sc_neurocore_engine as v3


class TestForwardNumpy:
    """Tests for single-call numpy dense forward."""

    def test_output_shape_and_type(self):
        layer = v3.DenseLayer(16, 8, 512)
        inputs = np.array([0.5] * 16, dtype=np.float64)
        out = layer.forward_numpy(inputs)
        assert isinstance(out, np.ndarray)
        assert out.shape == (8,)
        assert out.dtype == np.float64

    def test_output_range(self):
        layer = v3.DenseLayer(16, 8, 512)
        inputs = np.array([0.3] * 16, dtype=np.float64)
        out = layer.forward_numpy(inputs)
        assert np.all(out >= 0.0)
        assert np.all(out <= 16.0)

    def test_deterministic(self):
        layer = v3.DenseLayer(16, 8, 512, seed=42)
        inputs = np.array([0.5] * 16, dtype=np.float64)
        out1 = layer.forward_numpy(inputs, seed=100)
        out2 = layer.forward_numpy(inputs, seed=100)
        np.testing.assert_array_equal(out1, out2)

    def test_matches_forward_fast(self):
        """forward_numpy should match forward_fast with same seed."""
        layer = v3.DenseLayer(8, 4, 256, seed=42)
        inputs_list = [0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8]
        inputs_np = np.array(inputs_list, dtype=np.float64)
        out_fast = layer.forward_fast(inputs_list, seed=42)
        out_numpy = layer.forward_numpy(inputs_np, seed=42)
        np.testing.assert_allclose(out_numpy, out_fast)

    def test_wrong_input_length(self):
        layer = v3.DenseLayer(8, 4, 256)
        inputs = np.array([0.5] * 7, dtype=np.float64)
        with pytest.raises(ValueError):
            layer.forward_numpy(inputs)

    def test_different_seed_different_output(self):
        layer = v3.DenseLayer(8, 4, 1024, seed=42)
        inputs = np.array([0.5] * 8, dtype=np.float64)
        out1 = layer.forward_numpy(inputs, seed=100)
        out2 = layer.forward_numpy(inputs, seed=200)
        assert not np.array_equal(out1, out2)


class TestParallelBatchEncodeNumpy:
    """Tests for parallel batch_encode_numpy."""

    def test_shape_and_dtype(self):
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

    def test_popcount_statistics(self):
        """Encoded bitstreams should have popcount proportional to probability."""
        probs = np.array([0.25, 0.75], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=10_000, seed=42)
        pc0 = sum(int(w).bit_count() for w in packed[0])
        pc1 = sum(int(w).bit_count() for w in packed[1])
        assert abs(pc0 / 10_000 - 0.25) < 0.03
        assert abs(pc1 / 10_000 - 0.75) < 0.03

    def test_pipeline_encode_then_forward(self):
        """batch_encode_numpy -> forward_prepacked remains valid."""
        layer = v3.DenseLayer(4, 2, 1024, seed=42)
        probs = np.array([0.3, 0.5, 0.7, 0.9], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=55)
        out = layer.forward_prepacked(packed)
        assert len(out) == 2
        assert all(0.0 <= v <= 4.0 for v in out)

    def test_empty_probs(self):
        probs = np.array([], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=64, seed=42)
        assert packed.shape[0] == 0


class TestPhase8Version:
    def test_version_is_3_6_0(self):
        assert v3.__version__ == "3.6.0"
