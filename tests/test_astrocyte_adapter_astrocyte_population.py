# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAstrocytePopulation from former test_astrocyte_adapter.py

"""Focused suite: TestAstrocytePopulation from former test_astrocyte_adapter.py."""

from __future__ import annotations

from tests.astrocyte_adapter_support import *  # noqa: F403


class TestAstrocytePopulation:
    """Integration tests for adapter wiring through Population."""

    def test_create_population(self) -> None:
        """Population accepts the adapter class and constructor parameters."""
        pop = Population(AstrocyteNeuron, n=5, params={"ca_threshold": 0.3})
        assert pop.n == 5

    def test_step_all(self) -> None:
        """Population step_all returns a fixed-width spike vector."""
        pop = Population(AstrocyteNeuron, n=3, params={"ca_threshold": 0.3})
        currents = np.array([0.0, 1.0, 5.0])
        spikes = pop.step_all(currents)
        assert spikes.shape == (3,)
        assert spikes.dtype == np.int8

    def test_voltages_property(self) -> None:
        """Population voltages expose per-neuron calcium pseudo-voltage."""
        pop = Population(AstrocyteNeuron, n=4)
        pop.step_all(np.zeros(4))
        assert pop.voltages.shape == (4,)

    def test_reset_all(self) -> None:
        """Population reset_all delegates to every wrapped astrocyte."""
        pop = Population(AstrocyteNeuron, n=3)
        pop.step_all(np.ones(3) * 10.0)
        pop.reset_all()
        for neuron in pop.neurons:
            assert neuron.ca == 0.05
