# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for viz.plots spike train visualization

"""Tests for viz.plots — one per plot function, verifying Axes return."""

from __future__ import annotations

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.monitor import SpikeMonitor, StateMonitor
from sc_neurocore.network.network import Network
from sc_neurocore.viz import plots


@pytest.fixture()
def small_network():
    """Build a small network, run it briefly, return (net, spike_mon, state_mon, proj)."""
    pop_a = Population("LapicqueNeuron", 10, label="A")
    pop_b = Population("LapicqueNeuron", 10, label="B")
    proj = Projection(pop_a, pop_b, weight=0.5, probability=0.5)
    sm = SpikeMonitor(pop_a)
    st = StateMonitor(pop_a, variables=["v"])
    net = Network(pop_a, pop_b, proj, sm, st)
    net.run(duration=0.05, dt=0.001)
    return net, sm, st, proj


class TestRasterPlot:
    def test_returns_axes(self, small_network):
        _, sm, _, _ = small_network
        ax = plots.raster_plot(sm)
        assert ax is not None
        assert hasattr(ax, "figure")


class TestVoltageTrace:
    def test_returns_axes(self, small_network):
        _, _, st, _ = small_network
        ax = plots.voltage_trace(st, neuron_ids=[0, 1])
        assert ax is not None


class TestFiringRatePlot:
    def test_returns_axes(self, small_network):
        _, sm, _, _ = small_network
        ax = plots.firing_rate_plot(sm, bin_ms=5)
        assert ax is not None


class TestISIHistogram:
    def test_returns_axes(self, small_network):
        _, sm, _, _ = small_network
        ax = plots.isi_histogram(sm, neuron_id=0, bins=20)
        assert ax is not None


class TestCrossCorrelogram:
    def test_returns_axes(self, small_network):
        _, sm, _, _ = small_network
        ax = plots.cross_correlogram(sm, 0, 1, max_lag_ms=10)
        assert ax is not None


class TestPopulationActivity:
    def test_returns_axes(self, small_network):
        _, sm, _, _ = small_network
        ax = plots.population_activity(sm, bin_ms=5)
        assert ax is not None


class TestPhasePortrait:
    def test_returns_axes(self, small_network):
        _, _, st, _ = small_network
        ax = plots.phase_portrait(st, var_x="v", var_y="v", neuron_id=0)
        assert ax is not None


class TestWeightMatrix:
    def test_returns_axes(self, small_network):
        _, _, _, proj = small_network
        ax = plots.weight_matrix(proj)
        assert ax is not None


class TestNetworkGraph:
    def test_returns_axes(self, small_network):
        net, _, _, _ = small_network
        ax = plots.network_graph(net)
        assert ax is not None


class TestPSDPlot:
    def test_returns_axes(self, small_network):
        _, sm, _, _ = small_network
        ax = plots.psd_plot(sm, neuron_id=0)
        assert ax is not None


class TestInstantaneousRate:
    def test_returns_axes(self, small_network):
        _, sm, _, _ = small_network
        ax = plots.instantaneous_rate_plot(sm, neuron_id=0, sigma_ms=10)
        assert ax is not None


class TestSpikeTrainComparison:
    def test_returns_axes(self):
        trains = [np.array([1, 5, 10]), np.array([2, 6, 11]), np.array([3, 7, 12])]
        ax = plots.spike_train_comparison(trains, labels=["A", "B", "C"])
        assert ax is not None
