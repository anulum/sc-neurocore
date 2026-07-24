# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStochasticQuantize from former test_federated_sc.py

"""Focused suite: TestStochasticQuantize from former test_federated_sc.py."""

from __future__ import annotations

from federated_sc_support import *  # noqa: F403


class TestStochasticQuantize:
    def test_unbiased(self):
        g = np.array([0.3, 0.7, 0.5])
        results = [
            stochastic_quantize(g, levels=4, rng=np.random.default_rng(i)) for i in range(1000)
        ]
        mean_q = np.mean(results, axis=0)
        for i in range(3):
            assert abs(mean_q[i] - g[i]) < 0.05

    def test_output_in_range(self):
        rng = np.random.default_rng(42)
        g = np.array([-1.0, 0.5, 2.0])
        q = stochastic_quantize(g, levels=8, rng=rng)
        assert q.min() >= g.min() - 0.01
        assert q.max() <= g.max() + 0.01

    def test_constant_gradient(self):
        rng = np.random.default_rng(42)
        g = np.array([0.5, 0.5, 0.5])
        q = stochastic_quantize(g, levels=4, rng=rng)
        np.testing.assert_array_almost_equal(q, g)
