# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAuditoryProcessing from former test_model_zoo.py

"""Focused suite: TestAuditoryProcessing from former test_model_zoo.py."""

from __future__ import annotations

import pytest
import time
from sc_neurocore.model_zoo import (
    auditory_processing,
)
from sc_neurocore.network.network import (
    Network,
)
from sc_neurocore.neurons.models import (
    HodgkinHuxleyNeuron,
    WangBuzsakiNeuron,
)
from tests.model_zoo_support import (
    _run_and_count,
    _total_neurons,
)
from tests.performance_guard import assert_load_tolerant_throughput


class TestAuditoryProcessing:
    """Cochlear→onset→integration spectro-temporal SNN."""

    def test_returns_network(self) -> None:
        """The model returns network."""
        assert isinstance(auditory_processing(n_channels=8), Network)

    def test_three_populations(self) -> None:
        """cochlear, onset, integration."""
        net = auditory_processing(n_channels=8)
        assert len(net.populations) == 3
        assert net.populations[0].n == 8  # cochlear
        assert net.populations[1].n == 8  # onset
        assert net.populations[2].n == 4  # integration (n_channels // 2)

    def test_cochlear_uses_hh(self) -> None:
        """The cochlear population uses the Hodgkin-Huxley neuron."""
        net = auditory_processing(n_channels=8)
        assert net.populations[0]._model_cls is HodgkinHuxleyNeuron

    def test_onset_uses_wang_buzsaki(self) -> None:
        """Onset cells modelled as fast-spiking WangBuzsaki."""
        net = auditory_processing(n_channels=8)
        assert net.populations[1]._model_cls is WangBuzsakiNeuron

    def test_three_projections(self) -> None:
        """cochlear→onset, onset→onset (lateral inh), onset→integration."""
        net = auditory_processing(n_channels=8)
        assert len(net.projections) == 3

    def test_lateral_inhibition_negative(self) -> None:
        """onset→onset is inhibitory (weight=-2.0)."""
        net = auditory_processing(n_channels=8)
        onset_onset = net.projections[1]
        assert onset_onset.weight == -2.0

    def test_two_monitors(self) -> None:
        """The configuration attaches two spike monitors."""
        net = auditory_processing(n_channels=8)
        assert len(net.spike_monitors) == 2

    def test_produces_spikes(self) -> None:
        """The model produces spikes."""
        assert _run_and_count(auditory_processing(n_channels=8)) > 0

    @pytest.mark.parametrize("n_ch", [4, 8, 16])
    def test_scales_channels(self, n_ch: int) -> None:
        """The model scales channels."""
        net = auditory_processing(n_channels=n_ch)
        assert net.populations[0].n == n_ch
        assert net.populations[2].n == max(1, n_ch // 2)

    def test_performance(self) -> None:
        """The configuration simulates 50 ms above its load-tolerant throughput floor."""
        net = auditory_processing(n_channels=8)
        n_neurons = _total_neurons(net)
        t0 = time.perf_counter()
        net.run(0.05, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert_load_tolerant_throughput(
            label="auditory model-zoo network",
            observed_per_second=n_neurons * 50 / elapsed,
            strict_minimum_per_second=10.0,
        )
