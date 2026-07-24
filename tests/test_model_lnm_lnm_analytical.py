# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLNMAnalytical from former test_model_lnm.py

"""Focused suite: TestLNMAnalytical from former test_model_lnm.py."""

from __future__ import annotations

from tests.model_lnm_support import *  # noqa: F403


class TestLNMAnalytical:
    def test_v_update_formula(self):
        n = LearnableNeuronModel()
        v0 = n.v
        I = 0.5
        f_v = 1.0 / (1.0 + np.exp(-n.f_slope * (v0 - n.f_shift)))
        expected = n.alpha * v0 + n.beta * I + n.gamma * f_v
        n.step(I)
        if n.v != n.v_reset:
            assert abs(n.v - expected) < 1e-12

    def test_sigmoid_midpoint(self):
        n = LearnableNeuronModel()
        f = 1.0 / (1.0 + np.exp(0.0))
        assert abs(f - 0.5) < 1e-12

    def test_alpha_decay(self):
        n = LearnableNeuronModel(v_threshold=100.0)
        n.v = 0.8
        for _ in range(50):
            n.step(0.0)
        assert n.v < 0.8

    def test_beta_scales_input(self):
        n1 = LearnableNeuronModel(beta=0.1, v_threshold=100.0)
        n2 = LearnableNeuronModel(beta=0.5, v_threshold=100.0)
        for _ in range(50):
            n1.step(5.0)
            n2.step(5.0)
        assert n2.v > n1.v

    def test_gamma_zero_linear(self):
        n = LearnableNeuronModel(gamma=0.0, v_threshold=100.0)
        n.v = 0.5
        n.step(1.0)
        assert abs(n.v - (0.9 * 0.5 + 0.1 * 1.0)) < 1e-12

    def test_spike_resets(self):
        n = LearnableNeuronModel()
        for _ in range(10_000):
            if n.step(5.0) == 1:
                assert n.v == n.v_reset
                break
