# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
from __future__ import annotations
import numpy as np
from sc_neurocore.distillation import TemporalDistillationLoss, SelfDistiller


class TestTemporalDistillationLoss:
    def test_compute(self):
        loss_fn = TemporalDistillationLoss(temperature=3.0, alpha=0.5)
        result = loss_fn.compute(np.random.randn(10), np.random.randn(10))
        assert "total_loss" in result

    def test_with_targets(self):
        loss_fn = TemporalDistillationLoss()
        targets = np.zeros(5)
        targets[2] = 1.0
        result = loss_fn.compute(np.random.randn(5), np.random.randn(5), targets)
        assert result["task_loss"] > 0

    def test_temporal(self):
        loss_fn = TemporalDistillationLoss()
        result = loss_fn.compute(np.random.randn(8, 5), np.random.randn(8, 5))
        assert result["distill_loss"] >= 0


class TestSelfDistiller:
    def test_generate_targets(self):
        sd = SelfDistiller(T_teacher=16, T_student=4)
        targets = sd.generate_targets(lambda x, T: np.random.randn(10), np.zeros(8))
        assert targets.shape == (10,)
        assert abs(targets.sum() - 1.0) < 1e-6
