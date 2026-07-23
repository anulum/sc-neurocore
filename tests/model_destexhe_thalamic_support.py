# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_destexhe_thalamic.py

from __future__ import annotations

"""Full pipeline test for DestexheThalamicNeuron (Destexhe 1993).

Thalamocortical relay neuron with T-type calcium current:
4 ionic currents: I_Na(g=100, m³_inf·h), I_K(g=10, n⁴),
I_T(g=2, m²_T·h_T, Ca-mediated), I_L(g=0.05).

5 state variables: v, h_na, n_k, m_t(instantaneous), h_t.
5 sub-steps per call (dt=0.02). T-current enables:
- Tonic firing: depolarised, h_T inactivated
- Burst firing: from hyperpolarised state, h_T de-inactivated

Signature thalamic dynamics: rebound bursts after inhibition.
FULL PIPELINE WIRED + PERFORMANCE."""
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.destexhe_thalamic import DestexheThalamicNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi
def _run(neuron: DestexheThalamicNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]

__all__ = ['time', 'np', 'pytest', 'DestexheThalamicNeuron', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', 'firing_rate', 'isi', '_run']
