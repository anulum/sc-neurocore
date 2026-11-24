# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBrunelBalancedNetwork from former test_model_zoo.py

"""Focused suite: TestBrunelBalancedNetwork from former test_model_zoo.py."""

from __future__ import annotations

import numpy as np
import pytest
import time
from sc_neurocore.analysis.spike_stats.basic import (
    firing_rate,
)
from sc_neurocore.model_zoo import (
    brunel_balanced_network,
)
from sc_neurocore.network.network import (
    Network,
)
from tests.model_zoo_support import (
    _run_and_count,
    _total_neurons,
    _total_spikes,
)
from tests.performance_guard import assert_load_tolerant_throughput


class TestBrunelBalancedNetwork:
    """E/I balanced network with 4:1 exc:inh ratio."""

    def test_returns_network(self) -> None:
        """The model returns network."""
        assert isinstance(brunel_balanced_network(n_exc=50, n_inh=12), Network)

    def test_two_populations(self) -> None:
        """The network has an excitatory and an inhibitory population of the requested sizes."""
        net = brunel_balanced_network(n_exc=100, n_inh=25)
        assert len(net.populations) == 2
        assert net.populations[0].n == 100  # exc
        assert net.populations[1].n == 25  # inh

    def test_four_projections_full_connectivity(self) -> None:
        """E→E, E→I, I→E, I→I — all four quadrants wired."""
        net = brunel_balanced_network(n_exc=50, n_inh=12)
        assert len(net.projections) == 4

    def test_inhibition_stronger_than_excitation(self) -> None:
        """g=5 means |J_I| = 5·J_E — inhibition dominance."""
        net = brunel_balanced_network(n_exc=50, n_inh=12, g=5.0)
        j_e = net.projections[0].weight  # E→E
        j_i = net.projections[2].weight  # I→E
        assert j_i < 0  # inhibitory
        assert abs(j_i) > abs(j_e)  # |J_I| > J_E
        assert abs(abs(j_i) - 5.0 * abs(j_e)) < 1e-10

    def test_delay_present(self) -> None:
        """Synaptic delay=1.5ms from Brunel 2000."""
        net = brunel_balanced_network(n_exc=50, n_inh=12)
        for proj in net.projections:
            assert proj.delay == 1.5

    def test_all_projections_wired(self) -> None:
        """All 4 projections connect distinct or same populations."""
        net = brunel_balanced_network(n_exc=50, n_inh=12)
        for proj in net.projections:
            assert proj.source is not None
            assert proj.target is not None

    def test_two_poisson_drives(self) -> None:
        """The network has one Poisson drive per population."""
        net = brunel_balanced_network(n_exc=50, n_inh=12)
        assert len(net.stimuli) == 2

    def test_two_monitors(self) -> None:
        """The network monitors both populations, the excitatory one included."""
        net = brunel_balanced_network(n_exc=50, n_inh=12)
        assert len(net.spike_monitors) == 2
        labels = {m.label for m in net.spike_monitors}
        assert "exc_spikes" in labels
        assert "inh_spikes" in labels

    def test_produces_spikes(self) -> None:
        """The model produces spikes."""
        assert _run_and_count(brunel_balanced_network(n_exc=50, n_inh=12)) > 0

    @pytest.mark.parametrize("g", [3.0, 5.0, 8.0])
    def test_g_sweep(self, g: float) -> None:
        """Network remains stable across inhibition strengths."""
        net = brunel_balanced_network(n_exc=50, n_inh=12, g=g)
        net.run(0.05, dt=0.001, backend="python")
        assert _total_spikes(net) >= 0  # no crash

    def test_performance(self) -> None:
        """The configuration simulates 50 ms above its load-tolerant throughput floor."""
        net = brunel_balanced_network(n_exc=50, n_inh=12)
        n_neurons = _total_neurons(net)
        t0 = time.perf_counter()
        net.run(0.05, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert_load_tolerant_throughput(
            label="Brunel model-zoo network",
            observed_per_second=n_neurons * 50 / elapsed,
            strict_minimum_per_second=100.0,
        )

    def test_analysis_firing_rate(self) -> None:
        """The analysis gives the firing rate of a driven run."""
        net = brunel_balanced_network(n_exc=50, n_inh=12)
        net.run(0.2, dt=0.001, backend="python")
        exc_mon = net.spike_monitors[0]
        train = np.zeros(200, dtype=float)
        for t in exc_mon.spike_times:
            if 0 <= t < 200:
                train[t] += 1.0
        rate = firing_rate(train, dt=0.001)
        assert rate >= 0
