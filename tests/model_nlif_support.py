# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_nlif.py

from __future__ import annotations

"""Full pipeline test for NonlinearLIFNeuron (Touboul & Brette 2008).

Nonlinear LIF with quadratic/cubic term + adaptation:
C dV/dt = a·(V-V_rest)·(V-V_crit) - w + I
dw/dt = (b·(V-V_rest) - w) / tau_w

Quadratic nonlinearity a·(V-V_rest)·(V-V_crit) creates:
- Stable resting point at V_rest when I < rheobase
- Runaway depolarisation when V > V_crit (positive feedback)
- Hard threshold reset at V ≥ V_threshold

w provides spike-frequency adaptation (tau_w=100ms).
a=0.04, b=0.5, V_rest=-65, V_crit=-40.
FULL PIPELINE WIRED + PERFORMANCE."""
import os
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.nlif import NonlinearLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi
def _run(neuron: NonlinearLIFNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]

__all__ = ['os', 'time', 'np', 'pytest', 'NonlinearLIFNeuron', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', 'firing_rate', 'isi', '_run']
