# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_hindmarsh_rose.py

from __future__ import annotations

"""Pipeline test for HindmarshRoseNeuron (Hindmarsh & Rose 1984).

3D chaotic bursting model:
dx/dt = y - x³ + b·x² - z + I
dy/dt = 1 - 5·x² - y
dz/dt = r·(s·(x - x_rest) - z)

x: fast membrane-like variable. y: fast recovery.
z: slow adaptation (r=0.001) — modulates bursting.
b=3: controls burst width. s=4: z-x coupling.
Chaotic regime at intermediate I. Bursting at I≈3-5.
Default RK4 integration prioritizes trajectory fidelity over Euler throughput.
Pipeline and performance contract tests live in this module-specific file."""
import os
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.hindmarsh_rose import HindmarshRoseNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi
def _run(neuron: HindmarshRoseNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]

__all__ = ['os', 'time', 'np', 'pytest', 'HindmarshRoseNeuron', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', 'firing_rate', 'isi', '_run']
