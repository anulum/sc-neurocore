# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for astrocyte adapter (Population integration)

"""Tests for AstrocyteNeuron adapter wiring into Population/Network."""

import numpy as np

from sc_neurocore.neurons.models.astrocyte_adapter import AstrocyteNeuron
from sc_neurocore.network.population import Population


class TestAstrocyteNeuron:
    def test_step_returns_int(self):
        neuron = AstrocyteNeuron()
        result = neuron.step(0.0)
        assert result in (0, 1)

    def test_v_tracks_ca(self):
        neuron = AstrocyteNeuron()
        neuron.step(1.0)
        assert neuron.v == neuron.ca

    def test_reset(self):
        neuron = AstrocyteNeuron()
        for _ in range(50):
            neuron.step(5.0)
        neuron.reset()
        assert neuron.ca == 0.05
        assert neuron.v == 0.05

    def test_ip3_accessible(self):
        neuron = AstrocyteNeuron()
        assert neuron.ip3 > 0


class TestAstrocytePopulation:
    def test_create_population(self):
        pop = Population(AstrocyteNeuron, n=5, params={"ca_threshold": 0.3})
        assert pop.n == 5

    def test_step_all(self):
        pop = Population(AstrocyteNeuron, n=3, params={"ca_threshold": 0.3})
        currents = np.array([0.0, 1.0, 5.0])
        spikes = pop.step_all(currents)
        assert spikes.shape == (3,)
        assert spikes.dtype == np.int8

    def test_voltages_property(self):
        pop = Population(AstrocyteNeuron, n=4)
        pop.step_all(np.zeros(4))
        assert pop.voltages.shape == (4,)

    def test_reset_all(self):
        pop = Population(AstrocyteNeuron, n=3)
        pop.step_all(np.ones(3) * 10.0)
        pop.reset_all()
        for neuron in pop.neurons:
            assert neuron.ca == 0.05
