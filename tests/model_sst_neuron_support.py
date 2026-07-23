# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_sst_neuron.py

from __future__ import annotations

"""Full pipeline test for SSTNeuron (Pospischil 2008 LTS parameterisation).

Seven-state (V, m, h, n, p, s, r) somatostatin low-threshold spiking interneuron
integrated with candidate-first RK4. Includes a regression for the β_m offset:
the earlier ``-17`` numerator (shared with α_h) drove the cell into depolarisation
block — exactly three spikes then a fixed point near threshold for any stimulus —
while the published ``-40`` offset restores a monotone frequency-current relation."""
import numpy as np
import pytest
from sc_neurocore.neurons.models.sst_neuron import SSTNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
def _spikes(neuron: SSTNeuron, current: float, steps: int) -> int:
    return sum(neuron.step(current) for _ in range(steps))

__all__ = ['np', 'pytest', 'SSTNeuron', 'Population', 'Network', 'SpikeMonitor', 'PoissonInput', '_spikes']
