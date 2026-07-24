# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestForwardFast from former test_dense_optimization.py

"""Focused suite: TestForwardFast from former test_dense_optimization.py."""

from __future__ import annotations

from tests.dense_optimization_support import *  # noqa: F403


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
