# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_pv_fast_spiking_neuron.py

from __future__ import annotations

"""Full pipeline test for PVFastSpikingNeuron (Wang-Buzsáki 1996 + Kv3.1).

Parvalbumin fast-spiking interneuron: high-frequency, non-adapting discharge
sharpened by the Kv3.1 current. The candidate-first RK4 integrator advances the
four-state ``(V, h, n, p)`` system."""
import numpy as np
import pytest
from sc_neurocore.neurons.models.pv_fast_spiking_neuron import (
    PVFastSpikingNeuron,
    _safe_rate,
)
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
def _spikes(neuron: PVFastSpikingNeuron, current: float, steps: int) -> int:
    return sum(neuron.step(current) for _ in range(steps))

__all__ = ['np', 'pytest', 'PVFastSpikingNeuron', '_safe_rate', 'Population', 'Network', 'SpikeMonitor', 'PoissonInput', '_spikes']
