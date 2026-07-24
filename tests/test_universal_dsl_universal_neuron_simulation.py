# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestUniversalNeuronSimulation from former test_universal_dsl.py

"""Focused suite: TestUniversalNeuronSimulation from former test_universal_dsl.py."""

from __future__ import annotations

from tests.universal_dsl_support import *  # noqa: F403


class TestUniversalNeuronSimulation:
    """Test that UniversalNeuron produces physically reasonable dynamics."""

    def test_lif_spikes(self) -> None:
        neuron = UniversalNeuron.from_schema("lif")
        spikes = sum(neuron.step(I=30.0) for _ in range(200))
        assert spikes > 0, "LIF should spike with strong input"

    def test_lif_no_spike_without_input(self) -> None:
        neuron = UniversalNeuron.from_schema("lif")
        spikes = sum(neuron.step(I=0.0) for _ in range(200))
        assert spikes == 0, "LIF should not spike without input"

    def test_fitzhugh_nagumo_oscillates(self) -> None:
        neuron = UniversalNeuron.from_schema("fitzhugh_nagumo")
        spikes = sum(neuron.step(I=0.5) for _ in range(2000))
        assert spikes > 0, "FHN should oscillate with I=0.5"

    def test_izhikevich_spikes(self) -> None:
        neuron = UniversalNeuron.from_schema("izhikevich")
        spikes = sum(neuron.step(I=10.0) for _ in range(200))
        assert spikes > 0, "Izhikevich should spike with I=10"

    def test_hindmarsh_rose_evolves(self) -> None:
        neuron = UniversalNeuron.from_schema("hindmarsh_rose")
        initial_x = neuron.state["x"]
        for _ in range(500):
            neuron.step(I=3.0)
        assert neuron.state["x"] != initial_x, "HR should evolve from initial state"

    def test_adex_spikes(self) -> None:
        neuron = UniversalNeuron.from_schema("adex")
        spikes = sum(neuron.step(I=500.0) for _ in range(500))
        assert spikes > 0, "AdEx should spike with strong current"

    def test_escape_rate_schema_matches_hand_model_events_state_and_rng(self) -> None:
        from sc_neurocore.neurons.models.escape_rate import EscapeRateNeuron

        seed = 0xBEEF
        hand = EscapeRateNeuron(seed=seed)
        schema = UniversalNeuron.from_schema("escape_rate", rng_seed_override=seed)
        hand_events: list[int] = []
        schema_events: list[int] = []
        hand_trace: list[float] = []
        schema_trace: list[float] = []
        for _ in range(4096):
            hand_events.append(hand.step(17.0))
            schema_events.append(schema.step(I=17.0))
            hand_trace.append(hand.v)
            schema_trace.append(schema.state["v"])

        assert schema_events == hand_events
        np.testing.assert_allclose(schema_trace, hand_trace, rtol=0.0, atol=1.0e-14)
        equation = schema.to_equation_neuron()
        assert equation.escape_rng_initial_seed == seed
        assert equation.escape_rng_state == hand.rng_state
