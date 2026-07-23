# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_martinotti_neuron.py

from __future__ import annotations

"""Full pipeline test for MartinottiNeuron (Pospischil 2008 adapting interneuron).

Six-state (V, m, h, n, p, s) Martinotti cell integrated with candidate-first RK4.
Includes a regression for the β_m offset: the earlier ``-17`` numerator (shared
with α_h) drove the cell into depolarisation block — two or three spikes then a
fixed point near threshold for any stimulus — while the published ``-40`` offset
restores a monotone frequency-current relation under the strong M-current."""
import numpy as np
import pytest
from sc_neurocore.neurons.models.martinotti_neuron import MartinottiNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
def _spikes(neuron: MartinottiNeuron, current: float, steps: int) -> int:
    return sum(neuron.step(current) for _ in range(steps))

__all__ = ['np', 'pytest', 'MartinottiNeuron', 'Population', 'Network', 'SpikeMonitor', 'PoissonInput', '_spikes']
