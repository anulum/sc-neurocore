# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_ilif.py

from __future__ import annotations

"""Full pipeline test for InhibitoryLIFNeuron (2025).

LIF with temporal inhibitory mechanism:
inh_trace *= alpha_inh
V = alpha_m · V + I - inh_strength · inh_trace
Spike: V→V_reset, inh_trace += 1.

alpha_m = exp(-dt/tau_m), alpha_inh = exp(-dt/tau_inh).
Precomputed decay constants. Inhibitory trace creates temporal
suppression after each spike, shaping temporal coding.
FULL PIPELINE WIRED + PERFORMANCE."""
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.ilif import InhibitoryLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: InhibitoryLIFNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


__all__ = [
    "time",
    "np",
    "pytest",
    "InhibitoryLIFNeuron",
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
