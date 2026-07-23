# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_srm0.py

from __future__ import annotations

"""Full pipeline test for SRM0Neuron (Gerstner & Kistler 2002).

SRM zeroth order: LIF-like integration + refractory kernel eta.
eta decays exp(-dt/tau_eta) and provides afterhyperpolarisation.
Unlike SpikeResponseNeuron, this has actual voltage accumulation.
The maintained implementation uses the exact coupled linear flow for the
membrane and refractory kernel under constant current."""
import math
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.srm0 import SRM0Neuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, isi, firing_rate
def _run(neuron: SRM0Neuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]
def _exact_reference(neuron: SRM0Neuron, current: float) -> tuple[float, float]:
    membrane_decay = math.exp(-neuron.dt / neuron.tau_m)
    eta_decay = math.exp(-neuron.dt / neuron.tau_eta)
    rate_delta = (1.0 / neuron.tau_m) - (1.0 / neuron.tau_eta)
    if abs(rate_delta) < 1.0e-14:
        eta_coupling = neuron.dt * membrane_decay / neuron.tau_m
    else:
        eta_coupling = (eta_decay - membrane_decay) / (neuron.tau_m * rate_delta)
    steady = neuron.v_rest + neuron.resistance * current
    return (
        steady + (neuron.v - steady) * membrane_decay + neuron._eta * eta_coupling,
        neuron._eta * eta_decay,
    )

__all__ = ['math', 'time', 'np', 'pytest', 'SRM0Neuron', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', 'isi', 'firing_rate', '_run', '_exact_reference']
