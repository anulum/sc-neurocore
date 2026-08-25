# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC stochastic rate-adaptation network tests

"""Real network-path tests for the retained stochastic adaptation model."""

from __future__ import annotations

from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.network import Network
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.neurons.models.sc_stochastic_rate_adaptation import (
    SCStochasticRateAdaptationNeuron,
)


def test_population_exposes_sc_identity() -> None:
    population = Population(SCStochasticRateAdaptationNeuron, n=10, label="sc-sra")
    assert population.n == 10
    assert population.model_name == "SCStochasticRateAdaptationNeuron"


def test_network_produces_spikes() -> None:
    population = Population(SCStochasticRateAdaptationNeuron, n=20, label="sc-sra")
    drive = PoissonInput(n=20, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
    monitor = SpikeMonitor(population)
    Network(population, drive, monitor).run(duration=2.0, dt=0.001, backend="python")
    assert monitor.count > 0


def test_recurrent_projection_executes() -> None:
    population = Population(SCStochasticRateAdaptationNeuron, n=20, label="sc-sra")
    projection = Projection(population, population, weight=5.0, probability=0.2, seed=42)
    drive = PoissonInput(n=20, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
    monitor = SpikeMonitor(population)
    Network(population, projection, drive, monitor).run(duration=1.0, dt=0.001, backend="python")
    assert isinstance(monitor.spike_trains, dict)
