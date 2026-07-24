# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPVFastSpikingIsolation from former test_model_pv_fast_spiking_neuron.py

"""Focused suite: TestPVFastSpikingIsolation from former test_model_pv_fast_spiking_neuron.py."""

from __future__ import annotations

from tests.model_pv_fast_spiking_neuron_support import *  # noqa: F403


class TestPVFastSpikingIsolation:
    def test_construction_defaults(self):
        n = PVFastSpikingNeuron()
        assert n.v == -65.0
        assert n.g_kv3 == 5.0
        assert n.phi == 5.0
        assert n.dt == 0.01

    def test_step_returns_binary(self):
        assert PVFastSpikingNeuron().step(2.0) in (0, 1)

    def test_quiescent_without_drive(self):
        assert _spikes(PVFastSpikingNeuron(), 0.0, 20000) == 0

    def test_suprathreshold_high_frequency_firing(self):
        # The defining FS feature: sustained high-rate discharge under drive.
        assert _spikes(PVFastSpikingNeuron(), 2.0, 40000) >= 200

    def test_rate_increases_with_current(self):
        s1 = _spikes(PVFastSpikingNeuron(), 1.0, 30000)
        s2 = _spikes(PVFastSpikingNeuron(), 3.0, 30000)
        assert s1 < s2

    def test_no_spike_frequency_adaptation(self):
        n = PVFastSpikingNeuron()
        spike_times = [t for t in range(40000) if n.step(2.0)]
        assert len(spike_times) >= 20
        intervals = np.diff(spike_times)
        early = float(np.mean(intervals[:5]))
        late = float(np.mean(intervals[-5:]))
        # FS cells do not adapt: late inter-spike intervals stay close to early.
        assert late < early * 1.3

    def test_state_finite_long_run(self):
        n = PVFastSpikingNeuron()
        for _ in range(50000):
            n.step(2.0)
        for value in (n.v, n.h, n.n, n.p):
            assert np.isfinite(value)

    def test_reset_restores_initial(self):
        n = PVFastSpikingNeuron()
        for _ in range(1000):
            n.step(2.0)
        n.reset()
        assert n.v == -65.0
        assert (n.h, n.n, n.p) == (0.8, 0.1, 0.0)
