# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_psn.py

from __future__ import annotations

"""Full pipeline test for SCResettingParallelSpikingNeuron (PSN, 2024).

1D convolution kernel over circular buffer. Spike when
dot(kernel, buffer) >= threshold. Buffer cleared on spike."""
import numpy as np
import pytest
from sc_neurocore.neurons.models.sc_resetting_psn import SCResettingParallelSpikingNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi

__all__ = [
    "np",
    "pytest",
    "SCResettingParallelSpikingNeuron",
    "Population",
    "Projection",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "spike_count",
    "firing_rate",
    "isi",
]
