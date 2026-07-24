# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_hay_l5.py

from __future__ import annotations

"""Full pipeline test for HayL5PyramidalNeuron (Hay et al. 2011).

Reduced 3-compartment Layer 5 thick-tufted pyramidal cell:
Soma: I_Na(g=300, m³_inf·h), I_K(g=40, n⁴), I_L, coupling→trunk
Trunk: I_CaT(g=2, m²·h_ca), I_h(g=0.02, m_ih), I_L, coupling↔
Tuft: I_CaA(g=1.5, m²_inf), I_KCa(g=2.5, Ca-dep), I_L, coupling→trunk

9 state variables: v_s, h_na, n_k, v_t, m_ca, h_ca, m_ih, v_a, ca_a.
4 sub-steps (dt=0.025). Dual input: current_soma + current_tuft.
BAC firing: backpropagation-activated calcium spike in tuft.
Compartment areas: p_s=0.15, p_t=0.25, p_a=0.60.
FULL PIPELINE WIRED + PERFORMANCE."""
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.hay_l5 import HayL5PyramidalNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: HayL5PyramidalNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


__all__ = [
    "time",
    "np",
    "pytest",
    "HayL5PyramidalNeuron",
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
