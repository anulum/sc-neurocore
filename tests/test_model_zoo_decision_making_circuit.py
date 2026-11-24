# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDecisionMakingCircuit from former test_model_zoo.py

"""Focused suite: TestDecisionMakingCircuit from former test_model_zoo.py."""

from __future__ import annotations

import pytest
import time
from sc_neurocore.model_zoo import (
    decision_making_circuit,
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


class TestDecisionMakingCircuit:
    """Two competing pools + shared inhibition (attractor dynamics)."""

    def test_returns_network(self) -> None:
        """The model returns network."""
        assert isinstance(decision_making_circuit(n_per_pool=10), Network)

    def test_four_populations(self) -> None:
        """pool_A, pool_B, nonselective, inhibitory."""
        net = decision_making_circuit(n_per_pool=10)
        assert len(net.populations) == 4

    def test_pool_sizes(self) -> None:
        """The two selective pools and the nonselective pool have the requested sizes."""
        net = decision_making_circuit(n_per_pool=30)
        assert net.populations[0].n == 30  # pool_A
        assert net.populations[1].n == 30  # pool_B
        assert net.populations[2].n == max(10, 30 // 6)  # nonselective
        assert net.populations[3].n == max(15, 30 // 4)  # inhibitory

    def test_excitatory_uses_hh(self) -> None:
        """The excitatory pools use the Hodgkin-Huxley neuron."""
        net = decision_making_circuit(n_per_pool=10)
        for i in range(3):  # pools + nonselective
            assert net.populations[i]._model_cls is HodgkinHuxleyNeuron

    def test_inhibitory_uses_wang_buzsaki(self) -> None:
        """Inhibitory uses wang buzsaki."""
        net = decision_making_circuit(n_per_pool=10)
        assert net.populations[3]._model_cls is WangBuzsakiNeuron

    def test_nine_projections(self) -> None:
        """A→A, B→B, A→I, B→I, I→A, I→B, NS→A, NS→B, NS→I."""
        net = decision_making_circuit(n_per_pool=10)
        assert len(net.projections) == 9

    def test_potentiated_recurrent_excitation(self) -> None:
        """Within-pool recurrent weight=3.0 (potentiated)."""
        net = decision_making_circuit(n_per_pool=10)
        assert net.projections[0].weight == 3.0  # A→A
        assert net.projections[1].weight == 3.0  # B→B

    def test_cross_inhibition_negative(self) -> None:
        """I→A and I→B carry negative weight=-4.0."""
        net = decision_making_circuit(n_per_pool=10)
        assert net.projections[4].weight == -4.0  # I→A
        assert net.projections[5].weight == -4.0  # I→B

    def test_two_pool_monitors(self) -> None:
        """The circuit monitors both selective pools."""
        net = decision_making_circuit(n_per_pool=10)
        assert len(net.spike_monitors) == 2
        labels = {m.label for m in net.spike_monitors}
        assert "pool_A_spikes" in labels
        assert "pool_B_spikes" in labels

    def test_three_stimuli(self) -> None:
        """The circuit has three stimuli."""
        net = decision_making_circuit(n_per_pool=10)
        assert len(net.stimuli) == 3

    def test_produces_spikes(self) -> None:
        """The model produces spikes."""
        assert _run_and_count(decision_making_circuit(n_per_pool=10)) > 0

    @pytest.mark.parametrize("n_per_pool", [10, 30, 60])
    def test_scales_pool_size(self, n_per_pool: int) -> None:
        """The model scales pool size."""
        net = decision_making_circuit(n_per_pool=n_per_pool)
        assert net.populations[0].n == n_per_pool

    def test_performance(self) -> None:
        """The configuration simulates 50 ms above its load-tolerant throughput floor."""
        net = decision_making_circuit(n_per_pool=10)
        n_neurons = _total_neurons(net)
        t0 = time.perf_counter()
        net.run(0.05, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert_load_tolerant_throughput(
            label="decision model-zoo network",
            observed_per_second=n_neurons * 50 / elapsed,
            strict_minimum_per_second=10.0,
        )
