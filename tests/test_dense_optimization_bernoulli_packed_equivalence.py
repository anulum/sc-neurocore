# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBernoulliPackedEquivalence from former test_dense_optimization.py

"""Focused suite: TestBernoulliPackedEquivalence from former test_dense_optimization.py."""

from __future__ import annotations

from tests.dense_optimization_support import *  # noqa: F403

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
