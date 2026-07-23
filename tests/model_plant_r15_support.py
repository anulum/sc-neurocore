# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_plant_r15.py

from __future__ import annotations

"""Full pipeline test for PlantR15Neuron (Plant 1981, Aplysia R15).

5 ODEs: V, m, h, n, Ca. Parabolic burster with Ca-dependent K current.
At default parameters, model fires one transient spike then converges to
a stable equilibrium at V ≈ −23.8 mV (Ca accumulation suppresses firing)."""
import numpy as np
import pytest
from sc_neurocore.neurons.models.plant_r15 import PlantR15Neuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count
def _run(neuron: PlantR15Neuron, current: float, steps: int) -> tuple[list[int], list[float]]:
    """Return (spike_times, voltage_trace)."""
    spike_times: list[int] = []
    voltages: list[float] = []
    for t in range(steps):
        s = neuron.step(current)
        if s == 1:
            spike_times.append(t)
        voltages.append(neuron.v)
    return spike_times, voltages

__all__ = ['np', 'pytest', 'PlantR15Neuron', 'Population', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', '_run']
