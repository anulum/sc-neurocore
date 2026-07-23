# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCentralPatternGenerator from former test_model_zoo.py

"""Focused suite: TestCentralPatternGenerator from former test_model_zoo.py."""

from __future__ import annotations

from tests.model_zoo_support import *  # noqa: F403

class TestCentralPatternGenerator:
    """Half-centre CPG with mutually inhibiting oscillator pairs."""

    def test_returns_network(self):
        assert isinstance(central_pattern_generator(n_oscillators=2), Network)

    def test_population_count(self):
        """2 oscillators × 2 (flexor+extensor) × 5 neurons = 4 populations."""
        net = central_pattern_generator(n_oscillators=2)
        assert len(net.populations) == 4

    def test_uses_hindmarsh_rose(self):
        net = central_pattern_generator(n_oscillators=2)
        for pop in net.populations:
            assert pop._model_cls is HindmarshRoseNeuron

    def test_bursting_parameters(self):
        """b=3.0, r=0.005, s=4.0 — bursting regime."""
        net = central_pattern_generator(n_oscillators=2)
        neuron = net.populations[0].neurons[0]
        assert neuron.b == 3.0
        assert neuron.r == 0.005
        assert neuron.s == 4.0

    def test_mutual_inhibition_within_pair(self):
        """flex→ext and ext→flex are inhibitory (weight=-2.0)."""
        net = central_pattern_generator(n_oscillators=2)
        # First 2 projections per oscillator: flex→ext, ext→flex
        assert net.projections[0].weight == -2.0
        assert net.projections[1].weight == -2.0

    def test_inter_oscillator_excitatory(self):
        """Adjacent oscillators coupled with positive weight=1.0."""
        net = central_pattern_generator(n_oscillators=2)
        # Third projection per oscillator is inter-oscillator coupling
        assert net.projections[2].weight == 1.0

    def test_four_monitors(self):
        net = central_pattern_generator(n_oscillators=2)
        assert len(net.spike_monitors) == 4

    def test_produces_spikes(self):
        assert _run_and_count(central_pattern_generator(n_oscillators=2)) > 0

    @pytest.mark.parametrize("n_osc", [2, 3, 4])
    def test_scales_oscillators(self, n_osc: int):
        net = central_pattern_generator(n_oscillators=n_osc)
        assert len(net.populations) == 2 * n_osc

    def test_performance(self):
        net = central_pattern_generator(n_oscillators=2)
        n_neurons = _total_neurons(net)
        t0 = time.perf_counter()
        net.run(0.05, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert n_neurons * 50 / elapsed > 10
