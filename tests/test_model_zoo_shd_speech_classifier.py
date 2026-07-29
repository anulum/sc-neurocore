# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSHDSpeechClassifier from former test_model_zoo.py

"""Focused suite: TestSHDSpeechClassifier from former test_model_zoo.py."""

from __future__ import annotations

from tests.model_zoo_support import *  # noqa: F403
from tests.performance_guard import assert_load_tolerant_throughput


class TestSHDSpeechClassifier:
    """700-256-20 recurrent SNN for spiking Heidelberg digits."""

    def test_returns_network(self):
        assert isinstance(shd_speech_classifier(), Network)

    def test_topology_three_populations(self):
        net = shd_speech_classifier()
        assert len(net.populations) == 3
        assert net.populations[0].n == 700  # input
        assert net.populations[1].n == 256  # recurrent
        assert net.populations[2].n == 20  # output

    def test_three_projections_including_recurrent(self):
        """input→rec, rec→rec (recurrent), rec→output."""
        net = shd_speech_classifier()
        assert len(net.projections) == 3
        # Recurrent: source == target
        rec_proj = net.projections[1]
        assert rec_proj.source is rec_proj.target

    def test_recurrent_tau_longer(self):
        """Recurrent layer uses tau_mem=20 vs input tau_mem=10."""
        net = shd_speech_classifier()
        inp_neuron = net.populations[0].neurons[0]
        rec_neuron = net.populations[1].neurons[0]
        assert rec_neuron.tau_mem > inp_neuron.tau_mem

    def test_two_monitors(self):
        net = shd_speech_classifier()
        assert len(net.spike_monitors) == 2

    def test_produces_spikes(self):
        assert _run_and_count(shd_speech_classifier()) > 0

    def test_performance(self):
        net = shd_speech_classifier()
        n_neurons = _total_neurons(net)
        t0 = time.perf_counter()
        net.run(0.05, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert_load_tolerant_throughput(
            label="SHD model-zoo network",
            observed_per_second=n_neurons * 50 / elapsed,
            strict_minimum_per_second=100.0,
        )
