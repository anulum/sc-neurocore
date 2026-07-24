# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_viz_plots.py

from __future__ import annotations

"""Tests for viz.plots — one per plot function, verifying Axes return."""
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


__all__ = [
    "np",
    "pytest",
    "matplotlib",
    "Population",
    "Projection",
    "SpikeMonitor",
    "StateMonitor",
    "Network",
    "plots",
    "small_network",
    "__all__",
]
