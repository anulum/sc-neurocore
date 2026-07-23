# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTemporalDistillationLoss from former test_distillation.py

"""Focused suite: TestTemporalDistillationLoss from former test_distillation.py."""

from __future__ import annotations

from tests.distillation_support import *  # noqa: F403

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
