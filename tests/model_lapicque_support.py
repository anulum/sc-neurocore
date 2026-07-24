# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_lapicque.py

from __future__ import annotations

"""Full pipeline test for LapicqueNeuron (Lapicque 1907).

Classical RC integrate-and-fire — the original IF model:
τ · dV/dt = -(V - V_rest) + R·I
Spike: V → V_reset when V ≥ V_threshold.

Steady state: V_ss = V_rest + R·I. Fires only if V_ss ≥ V_threshold,
i.e. I ≥ (V_threshold - V_rest) / R = rheobase.
Exact constant-current flow:
V(t + dt) = V_ss + (V(t) - V_ss) · exp(-dt / τ).
"""
import os
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.lapicque import LapicqueNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: LapicqueNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


__all__ = [
    "os",
    "time",
    "np",
    "pytest",
    "LapicqueNeuron",
    "Population",
    "Projection",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "spike_count",
    "firing_rate",
    "isi",
    "_run",
]
