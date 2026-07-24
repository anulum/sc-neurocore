# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBatchEncodeNumpy from former test_dense_optimization.py

"""Focused suite: TestBatchEncodeNumpy from former test_dense_optimization.py."""

from __future__ import annotations

from tests.dense_optimization_support import *  # noqa: F403


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
