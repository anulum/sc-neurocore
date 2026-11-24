# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMNISTClassifier from former test_model_zoo.py

"""Focused suite: TestMNISTClassifier from former test_model_zoo.py."""

from __future__ import annotations

import numpy as np
import pytest
import time
from sc_neurocore.analysis.spike_stats.basic import (
    spike_count,
)
from sc_neurocore.model_zoo import (
    mnist_classifier,
)
from sc_neurocore.network.network import (
    Network,
)
from sc_neurocore.neurons import (
    StochasticLIFNeuron,
)
from tests.model_zoo_support import (
    _run_and_count,
    _total_neurons,
)
from tests.performance_guard import assert_load_tolerant_throughput


class TestMNISTClassifier:
    """784-128-10 feedforward SNN for digit classification."""

    def test_returns_network(self) -> None:
        """The model returns network."""
        net = mnist_classifier(n_hidden=32)
        assert isinstance(net, Network)

    def test_topology_three_populations(self) -> None:
        """Topology three populations."""
        net = mnist_classifier(n_hidden=64)
        assert len(net.populations) == 3
        assert net.populations[0].n == 784
        assert net.populations[1].n == 64
        assert net.populations[2].n == 10

    def test_two_feedforward_projections(self) -> None:
        """Two feedforward projections."""
        net = mnist_classifier(n_hidden=32)
        assert len(net.projections) == 2
        # input→hidden, hidden→output
        assert net.projections[0].source is net.populations[0]
        assert net.projections[0].target is net.populations[1]
        assert net.projections[1].source is net.populations[1]
        assert net.projections[1].target is net.populations[2]

    def test_two_spike_monitors(self) -> None:
        """The classifier attaches two spike monitors."""
        net = mnist_classifier(n_hidden=32)
        assert len(net.spike_monitors) == 2

    def test_uses_stochastic_lif(self) -> None:
        """The model uses stochastic LIF."""
        net = mnist_classifier(n_hidden=32)
        for pop in net.populations:
            assert pop._model_cls is StochasticLIFNeuron

    def test_xavier_weight_scaling(self) -> None:
        """Weight ∝ sqrt(2/fan_in) — Xavier initialisation."""
        net = mnist_classifier(n_hidden=128)
        expected_ih = np.sqrt(2.0 / 784) * 20.0
        assert abs(net.projections[0].weight - expected_ih) < 1e-10
        expected_ho = np.sqrt(2.0 / 128) * 20.0
        assert abs(net.projections[1].weight - expected_ho) < 1e-10

    def test_stimulus_attached(self) -> None:
        """The single stimulus drives the input population."""
        net = mnist_classifier(n_hidden=32)
        assert len(net.stimuli) == 1
        assert net.stimuli[0].target is net.populations[0]

    def test_produces_spikes(self) -> None:
        """The model produces spikes."""
        assert _run_and_count(mnist_classifier(n_hidden=32)) > 0

    def test_output_monitor_records(self) -> None:
        """Output monitor records."""
        net = mnist_classifier(n_hidden=32)
        net.run(0.1, dt=0.001, backend="python")
        # Output monitor is first (mon_out added before mon_hid)
        output_mon = net.spike_monitors[0]
        assert output_mon.label == "output_spikes"

    @pytest.mark.parametrize("n_hidden", [16, 64, 128])
    def test_scales_hidden_size(self, n_hidden: int) -> None:
        """The model scales hidden size."""
        net = mnist_classifier(n_hidden=n_hidden)
        assert net.populations[1].n == n_hidden

    def test_performance(self) -> None:
        """The configuration simulates 50 ms above its load-tolerant throughput floor."""
        net = mnist_classifier(n_hidden=16)
        n_neurons = _total_neurons(net)
        t0 = time.perf_counter()
        net.run(0.05, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        rate = n_neurons * 50 / elapsed
        assert_load_tolerant_throughput(
            label="MNIST model-zoo network",
            observed_per_second=rate,
            strict_minimum_per_second=100.0,
        )

    def test_analysis_spike_count(self) -> None:
        """The analysis counts the spikes of a driven run."""
        net = mnist_classifier(n_hidden=32)
        net.run(0.1, dt=0.001, backend="python")
        for mon in net.spike_monitors:
            times = mon.spike_times
            train = np.zeros(100, dtype=float)
            for t in times:
                if 0 <= t < 100:
                    train[t] = 1.0
            sc = spike_count(train)
            assert sc >= 0
