# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestContinualLearner from former test_continual.py

"""Focused suite: TestContinualLearner from former test_continual.py."""

from __future__ import annotations

from tests.continual_support import *  # noqa: F403


class TestContinualLearner:
    def _make_learner(self):
        weights = [np.random.randn(16, 8) * 0.3, np.random.randn(4, 16) * 0.3]
        return ContinualLearner(weights, layer_names=["hidden", "output"])

    def test_init(self):
        cl = self._make_learner()
        assert len(cl.weights) == 2
        assert cl.ewc_lambda == 1000.0

    def test_weights_are_copies(self):
        w = [np.ones((4, 4))]
        cl = ContinualLearner(w)
        w[0][0, 0] = 999.0
        assert cl.weights[0][0, 0] == 1.0

    def test_compute_fisher(self):
        cl = self._make_learner()
        grads = [[np.random.randn(16, 8), np.random.randn(4, 16)] for _ in range(10)]
        cl.compute_fisher(grads)
        assert cl._fisher_diag is not None
        assert cl._star_weights is not None
        assert all(f.shape == w.shape for f, w in zip(cl._fisher_diag, cl.weights))

    def test_ewc_penalty_before_fisher(self):
        cl = self._make_learner()
        assert cl.ewc_penalty() == 0.0

    def test_ewc_penalty_zero_when_unchanged(self):
        cl = self._make_learner()
        grads = [[np.random.randn(16, 8), np.random.randn(4, 16)] for _ in range(10)]
        cl.compute_fisher(grads)
        assert cl.ewc_penalty() < 1e-10

    def test_ewc_penalty_positive_after_weight_change(self):
        cl = self._make_learner()
        grads = [[np.random.randn(16, 8), np.random.randn(4, 16)] for _ in range(10)]
        cl.compute_fisher(grads)
        cl.update_weights([w + 0.1 for w in cl.weights])
        assert cl.ewc_penalty() > 0

    def test_register_task(self):
        cl = self._make_learner()
        cl.register_task(0.95)
        cl.register_task(0.92)
        assert cl._task_count == 2
        assert cl._accuracy_history == [0.95, 0.92]

    def test_extract_plasticity_configs(self):
        cl = self._make_learner()
        configs = cl.extract_plasticity_configs()
        assert len(configs) == 2
        assert configs[0].layer_name == "hidden"
        assert configs[0].rule == "stdp"
        assert configs[0].lr_potentiation > 0
        assert configs[0].lr_depression > configs[0].lr_potentiation

    def test_report(self):
        cl = self._make_learner()
        cl.register_task(0.95)
        r = cl.report()
        assert isinstance(r, ContinualReport)
        assert r.tasks_trained == 1
        assert len(r.plasticity_configs) == 2
        s = r.summary()
        assert "Continual Learning" in s
        assert "0.9500" in s

    def test_default_layer_names(self):
        weights = [np.random.randn(8, 4)]
        cl = ContinualLearner(weights)
        assert cl.layer_names == ["layer_0"]

    def test_update_weights(self):
        cl = self._make_learner()
        new_w = [np.ones_like(w) for w in cl.weights]
        cl.update_weights(new_w)
        np.testing.assert_array_equal(cl.weights[0], 1.0)
