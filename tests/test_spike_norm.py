# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

from __future__ import annotations
import numpy as np
from sc_neurocore.spike_norm import (
    ThresholdDependentBN,
    PerTimestepBN,
    TemporalEffectiveBN,
    MembranePotentialBN,
    TemporalAccumulatedBN,
)


class TestThresholdDependentBN:
    def test_forward(self):
        bn = ThresholdDependentBN(n_features=8, threshold=1.0)
        out = bn.forward(np.random.randn(4, 8), training=True)
        assert out.shape == (4, 8)

    def test_inference(self):
        bn = ThresholdDependentBN(n_features=4)
        bn.forward(np.random.randn(10, 4), training=True)
        out = bn.forward(np.random.randn(3, 4), training=False)
        assert out.shape == (3, 4)


class TestPerTimestepBN:
    def test_forward(self):
        bn = PerTimestepBN(n_features=8, T=10)
        out = bn.forward(np.random.randn(4, 8), t=3, training=True)
        assert out.shape == (4, 8)


class TestTemporalEffectiveBN:
    def test_forward(self):
        bn = TemporalEffectiveBN(n_features=8, T=10)
        out = bn.forward(np.random.randn(4, 8), t=5, training=True)
        assert out.shape == (4, 8)


class TestMembranePotentialBN:
    def test_forward(self):
        bn = MembranePotentialBN(n_features=8)
        out = bn.forward(np.random.randn(4, 8), training=True)
        assert out.shape == (4, 8)

    def test_fused_threshold(self):
        bn = MembranePotentialBN(n_features=4, threshold=1.0)
        bn.forward(np.random.randn(10, 4), training=True)
        fused = bn.fused_threshold()
        assert fused.shape == (4,)

    def test_inference_passthrough(self):
        bn = MembranePotentialBN(n_features=4)
        x = np.random.randn(3, 4)
        out = bn.forward(x, training=False)
        np.testing.assert_array_equal(out, x)


class TestTemporalAccumulatedBN:
    def test_forward(self):
        bn = TemporalAccumulatedBN(n_features=8)
        for _ in range(5):
            out = bn.forward(np.random.randn(4, 8), training=True)
        assert out.shape == (4, 8)

    def test_reset(self):
        bn = TemporalAccumulatedBN(n_features=4)
        bn.forward(np.ones(4), training=True)
        bn.reset()
        assert np.allclose(bn._accumulated, 0)
