# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestForwardPrepacked from former test_dense_optimization.py

"""Focused suite: TestForwardPrepacked from former test_dense_optimization.py."""

from __future__ import annotations

from tests.dense_optimization_support import *  # noqa: F403

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
