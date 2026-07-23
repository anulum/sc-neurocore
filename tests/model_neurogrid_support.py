# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_neurogrid.py

from __future__ import annotations

"""Full pipeline test for NeuroGridNeuron (Boahen 2014).

2-compartment analog neuromorphic neuron:
Dendrite: dv_d/dt = (-(v_d-v_rest) + I - g_c·(v_d-v_s)) / tau_d
Soma:     dv_s/dt = (-(v_s-v_rest) + Δ_T·exp((v_s-θ)/Δ_T) + g_c·(v_d-v_s)) / tau_s

Dendrite (tau_d=50ms) passively integrates synaptic input.
Soma (tau_s=20ms) has EIF exponential spike initiation (Δ_T=2mV).
Compartments coupled by conductance g_c=0.5.
On v_s ≥ v_peak(20): v_s → v_reset(-65). exp clipped at 20.
FULL PIPELINE WIRED + PERFORMANCE."""
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.neurogrid import NeuroGridNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi
from tests.performance_guard import assert_throughput_guard
def _run(neuron: NeuroGridNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]

__all__ = ['time', 'np', 'pytest', 'NeuroGridNeuron', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', 'firing_rate', 'isi', 'assert_throughput_guard', '_run']
