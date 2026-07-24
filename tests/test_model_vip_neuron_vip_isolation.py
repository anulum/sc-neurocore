# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestVIPIsolation from former test_model_vip_neuron.py

"""Focused suite: TestVIPIsolation from former test_model_vip_neuron.py."""

from __future__ import annotations

from tests.model_vip_neuron_support import *  # noqa: F403


class TestVIPIsolation:
    def test_construction_defaults(self):
        n = VIPNeuron()
        assert n.v == -65.0
        assert n.g_a == 8.0
        assert n.c_m == 0.5
        assert n.dt == 0.025

    def test_step_returns_binary(self):
        assert VIPNeuron().step(1.0) in (0, 1)

    def test_quiescent_without_drive(self):
        assert _spikes(VIPNeuron(), 0.0, 20000) == 0

    def test_suprathreshold_spiking(self):
        assert _spikes(VIPNeuron(), 1.0, 40000) >= 10

    def test_rate_increases_with_current(self):
        s1 = _spikes(VIPNeuron(), 0.6, 30000)
        s2 = _spikes(VIPNeuron(), 2.0, 30000)
        assert s1 < s2

    def test_state_finite_long_run(self):
        n = VIPNeuron()
        for _ in range(50000):
            n.step(1.0)
        for value in (n.v, n.h, n.n, n.a, n.b):
            assert np.isfinite(value)

    def test_reset_restores_initial(self):
        n = VIPNeuron()
        for _ in range(1000):
            n.step(1.0)
        n.reset()
        assert n.v == -65.0
        assert (n.h, n.n, n.a, n.b) == (0.8, 0.1, 0.0, 0.9)
