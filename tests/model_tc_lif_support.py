# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_tc_lif.py

from __future__ import annotations

"""Full pipeline test for TwoCompartmentLIFNeuron (Yang et al. AAAI 2024).

Soma + dendrite: dendritic input provides history-dependent sequential
context via kappa coupling. step(i_soma, i_dend). Performance: ~635K steps/s."""
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.tc_lif import TwoCompartmentLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(
    neuron: TwoCompartmentLIFNeuron, i_soma: float, steps: int, i_dend: float = 0.0
) -> list[int]:
    return [t for t in range(steps) if neuron.step(i_soma, i_dend) == 1]


__all__ = [
    "time",
    "np",
    "pytest",
    "TwoCompartmentLIFNeuron",
    "Population",
    "Projection",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "spike_count",
    "firing_rate",
    "_run",
]
