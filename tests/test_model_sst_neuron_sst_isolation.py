# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSSTIsolation from former test_model_sst_neuron.py

"""Focused suite: TestSSTIsolation from former test_model_sst_neuron.py."""

from __future__ import annotations

from tests.model_sst_neuron_support import *  # noqa: F403


class TestSSTIsolation:
    def test_construction_defaults(self):
        n = SSTNeuron()
        assert n.v == -65.0
        assert n.g_na == 50.0
        assert n.g_m == 0.12
        assert n.c_m == 1.0
        assert n.dt == 0.025

    def test_step_returns_binary(self):
        assert SSTNeuron().step(2.0) in (0, 1)

    def test_quiescent_without_drive(self):
        assert _spikes(SSTNeuron(), 0.0, 20000) == 0

    def test_suprathreshold_spiking(self):
        assert _spikes(SSTNeuron(), 2.0, 40000) > 20

    def test_state_finite_long_run(self):
        n = SSTNeuron()
        for _ in range(50000):
            n.step(2.0)
        for value in (n.v, n.m, n.h, n.n, n.p, n.s, n.r):
            assert np.isfinite(value)

    def test_reset_restores_initial(self):
        n = SSTNeuron()
        for _ in range(1000):
            n.step(2.0)
        n.reset()
        assert n.v == -65.0
        assert (n.m, n.h, n.n, n.p, n.s, n.r) == (0.02, 0.8, 0.2, 0.0, 0.9, 0.1)
