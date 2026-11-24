# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWorkingMemoryCircuit from former test_model_zoo.py

"""Focused suite: TestWorkingMemoryCircuit from former test_model_zoo.py."""

from __future__ import annotations

import pytest
import time
from sc_neurocore.model_zoo import (
    working_memory_circuit,
)
from sc_neurocore.network.network import (
    Network,
)
from sc_neurocore.neurons.models import (
    CompteWMNeuron,
    WangBuzsakiNeuron,
)
from tests.model_zoo_support import (
    _run_and_count,
    _total_neurons,
)
from tests.performance_guard import assert_load_tolerant_throughput


class TestWorkingMemoryCircuit:
    """Ring attractor with NMDA-based persistent activity."""

    def test_returns_network(self) -> None:
        """The model returns network."""
        assert isinstance(working_memory_circuit(n_neurons=50), Network)

    def test_two_populations(self) -> None:
        """80% excitatory, 20% inhibitory."""
        net = working_memory_circuit(n_neurons=50)
        assert len(net.populations) == 2
        assert net.populations[0].n == 40  # 80%
        assert net.populations[1].n == 10  # 20%

    def test_excitatory_uses_compte_wm(self) -> None:
        """Excitatory uses compte wm."""
        net = working_memory_circuit(n_neurons=50)
        assert net.populations[0]._model_cls is CompteWMNeuron

    def test_inhibitory_uses_wang_buzsaki(self) -> None:
        """Inhibitory uses wang buzsaki."""
        net = working_memory_circuit(n_neurons=50)
        assert net.populations[1]._model_cls is WangBuzsakiNeuron

    def test_nmda_parameters(self) -> None:
        """NMDA conductance from Compte et al. 2000."""
        net = working_memory_circuit(n_neurons=50)
        neuron = net.populations[0].neurons[0]
        assert neuron.g_nmda == 0.165
        assert neuron.tau_nmda == 100.0
        assert neuron.mg == 1.0

    def test_four_projections(self) -> None:
        """E→E (ring), E→I, I→E, I→I."""
        net = working_memory_circuit(n_neurons=50)
        assert len(net.projections) == 4

    def test_excitatory_recurrent_self_connection(self) -> None:
        """E→E projection: source and target are the same population."""
        net = working_memory_circuit(n_neurons=50)
        p_ee = net.projections[0]
        assert p_ee.source is p_ee.target  # recurrent ring

    def test_two_monitors(self) -> None:
        """The circuit attaches two spike monitors."""
        net = working_memory_circuit(n_neurons=50)
        assert len(net.spike_monitors) == 2

    def test_produces_spikes(self) -> None:
        """The model produces spikes."""
        assert _run_and_count(working_memory_circuit(n_neurons=50)) > 0

    @pytest.mark.parametrize("n_neurons", [50, 100, 200])
    def test_scales_neuron_count(self, n_neurons: int) -> None:
        """The model scales neuron count."""
        net = working_memory_circuit(n_neurons=n_neurons)
        n_exc = int(0.8 * n_neurons)
        n_inh = n_neurons - n_exc
        assert net.populations[0].n == n_exc
        assert net.populations[1].n == n_inh

    def test_performance(self) -> None:
        """The configuration simulates 50 ms above its load-tolerant throughput floor."""
        net = working_memory_circuit(n_neurons=50)
        n_neurons = _total_neurons(net)
        t0 = time.perf_counter()
        net.run(0.05, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert_load_tolerant_throughput(
            label="working-memory model-zoo network",
            observed_per_second=n_neurons * 50 / elapsed,
            strict_minimum_per_second=10.0,
        )
