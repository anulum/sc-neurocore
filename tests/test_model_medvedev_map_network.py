# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Medvedev network integration contract

"""Network population and monitor integration test for the Medvedev map."""

from __future__ import annotations

from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.network import Network
from sc_neurocore.network.population import Population
from sc_neurocore.neurons.models.medvedev_map import MedvedevMapNeuron


def test_population_network_path_observes_events() -> None:
    """The standard network loop can drive and monitor the renamed u state."""
    population = Population(MedvedevMapNeuron, n=4, label="medvedev")
    monitor = SpikeMonitor(population)
    network = Network(population, monitor)
    network.run(duration=0.01, dt=0.001, backend="python")
    assert monitor.count > 0
