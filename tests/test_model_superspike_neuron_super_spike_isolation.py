# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSuperSpikeIsolation from former test_model_superspike_neuron.py

"""Focused suite: TestSuperSpikeIsolation from former test_model_superspike_neuron.py."""

from __future__ import annotations

from tests.model_superspike_neuron_support import *  # noqa: F403


class TestSuperSpikeIsolation:
    def test_construction_defaults(self):
        n = SuperSpikeNeuron()
        assert n.v == 0.0
        assert n.trace == 0.0
        assert n.tau_m == 10.0
        assert n.tau_e == 10.0
        assert n.v_threshold == 1.0
        assert n.beta_sg == 10.0

    def test_alpha_precomputed(self):
        n = SuperSpikeNeuron()
        assert abs(n.alpha_m - np.exp(-1.0 / 10.0)) < 1e-12
        assert abs(n.alpha_e - np.exp(-1.0 / 10.0)) < 1e-12

    def test_step_returns_binary(self):
        assert SuperSpikeNeuron().step(0.0) in (0, 1)

    def test_state_finite(self):
        n = SuperSpikeNeuron()
        for _ in range(50000):
            n.step(0.2)
        assert np.isfinite(n.v) and np.isfinite(n.trace)

    def test_reset(self):
        n = SuperSpikeNeuron()
        for _ in range(100):
            n.step(0.5)
        n.reset()
        assert n.v == 0.0 and n.trace == 0.0
