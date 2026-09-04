# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMetaLearner from former test_advanced_plasticity.py

"""Focused suite: TestMetaLearner from former test_advanced_plasticity.py."""

from __future__ import annotations

from tests.advanced_plasticity_support import *  # noqa: F403


class TestMetaLearner:
    def test_inner_loop(self, simple_net):
        net, pop_a, _, proj = simple_net
        w_before = proj.data.copy()
        inputs = np.ones((20, pop_a.n)) * 50.0
        targets = np.ones((20, pop_a.n))
        ml = MetaLearner(net, inner_lr=0.1)
        ml.inner_loop((inputs, targets), n_steps=5)
        assert not np.allclose(proj.data, w_before)

    def test_outer_step(self, simple_net):
        net, pop_a, _, proj = simple_net
        tasks = [(np.random.randn(5, pop_a.n) * 5, np.zeros((5, pop_a.n))) for _ in range(3)]
        ml = MetaLearner(net, inner_lr=0.01, outer_lr=0.001)
        w_before = proj.data.copy()
        ml.outer_step(tasks)
        assert proj.data.shape == w_before.shape
