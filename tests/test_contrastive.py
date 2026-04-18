# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
from __future__ import annotations
import numpy as np
from sc_neurocore.contrastive import SpikeContrastiveLoss, CSDPRule


class TestSpikeContrastiveLoss:
    def test_compute(self):
        assert SpikeContrastiveLoss().compute(np.random.rand(8, 16), np.random.rand(8, 16)) > 0

    def test_identical(self):
        a = np.random.rand(4, 8)
        assert SpikeContrastiveLoss().compute(a, a) >= 0

    def test_single(self):
        assert SpikeContrastiveLoss().compute(np.random.rand(1, 4), np.random.rand(1, 4)) == 0.0


class TestCSDPRule:
    def test_positive(self):
        W = np.random.randn(4, 8)
        assert not np.allclose(CSDPRule(lr=0.01).positive_update(W, np.ones(8), np.ones(4)), W)

    def test_negative(self):
        W = np.random.randn(4, 8)
        assert not np.allclose(CSDPRule(lr=0.01).negative_update(W, np.ones(8), np.ones(4)), W)

    def test_contrastive_step(self):
        rule = CSDPRule(lr=0.01)
        W = np.random.randn(4, 8)
        assert (
            rule.contrastive_step(W, np.ones(8), np.ones(4), np.zeros(8), np.zeros(4)).shape
            == W.shape
        )

    def test_goodness(self):
        rule = CSDPRule()
        assert rule.goodness(np.ones(10)) > rule.goodness(np.zeros(10))
