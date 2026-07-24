# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_golomb_fs.py

from __future__ import annotations

"""Full pipeline test for GolombFSNeuron (Golomb et al. 2007).

Fast-spiking interneuron with Kv3 potassium channel:
4 currents: I_Na(g=112.5, m³_inf·h), I_Kd(g=225, n⁴),
I_Kv3(g=150, p²), I_L(g=0.25).

Kv3: high-threshold (v_half=-3), fast activation → narrow spikes,
minimal spike-frequency adaptation, sustained high-rate firing.
3 gating variables: h (Na inact), n (Kd), p (Kv3).
m_Na is instantaneous. 10 sub-steps per call (dt=0.01).
FULL PIPELINE WIRED + PERFORMANCE."""
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.golomb_fs import GolombFSNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: GolombFSNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


__all__ = [
    "time",
    "np",
    "pytest",
    "GolombFSNeuron",
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
