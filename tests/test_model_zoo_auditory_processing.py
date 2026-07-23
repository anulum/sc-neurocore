# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAuditoryProcessing from former test_model_zoo.py

"""Focused suite: TestAuditoryProcessing from former test_model_zoo.py."""

from __future__ import annotations

from tests.model_zoo_support import *  # noqa: F403

class TestAuditoryProcessing:
    """Cochlear→onset→integration spectro-temporal SNN."""

    def test_returns_network(self):
        assert isinstance(auditory_processing(n_channels=8), Network)

    def test_three_populations(self):
        """cochlear, onset, integration."""
        net = auditory_processing(n_channels=8)
        assert len(net.populations) == 3
        assert net.populations[0].n == 8  # cochlear
        assert net.populations[1].n == 8  # onset
        assert net.populations[2].n == 4  # integration (n_channels // 2)

    def test_cochlear_uses_hh(self):
        net = auditory_processing(n_channels=8)
        assert net.populations[0]._model_cls is HodgkinHuxleyNeuron

    def test_onset_uses_wang_buzsaki(self):
        """Onset cells modelled as fast-spiking WangBuzsaki."""
        net = auditory_processing(n_channels=8)
        assert net.populations[1]._model_cls is WangBuzsakiNeuron

    def test_three_projections(self):
        """cochlear→onset, onset→onset (lateral inh), onset→integration."""
        net = auditory_processing(n_channels=8)
        assert len(net.projections) == 3

    def test_lateral_inhibition_negative(self):
        """onset→onset is inhibitory (weight=-2.0)."""
        net = auditory_processing(n_channels=8)
        onset_onset = net.projections[1]
        assert onset_onset.weight == -2.0

    def test_two_monitors(self):
        net = auditory_processing(n_channels=8)
        assert len(net.spike_monitors) == 2

    def test_produces_spikes(self):
        assert _run_and_count(auditory_processing(n_channels=8)) > 0

    @pytest.mark.parametrize("n_ch", [4, 8, 16])
    def test_scales_channels(self, n_ch: int):
        net = auditory_processing(n_channels=n_ch)
        assert net.populations[0].n == n_ch
        assert net.populations[2].n == max(1, n_ch // 2)

    def test_performance(self):
        net = auditory_processing(n_channels=8)
        n_neurons = _total_neurons(net)
        t0 = time.perf_counter()
        net.run(0.05, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert n_neurons * 50 / elapsed > 10
