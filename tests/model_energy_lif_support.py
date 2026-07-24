# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_energy_lif.py

from __future__ import annotations

"""Full pipeline test for EnergyLIFNeuron exact-flow hardening.

LIF with metabolic energy constraint ε. Spike cost depletes ε."""
import numpy as np
import pytest
from sc_neurocore.neurons.models.energy_lif import EnergyLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count

__all__ = [
    "np",
    "pytest",
    "EnergyLIFNeuron",
    "Population",
    "Projection",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "firing_rate",
    "spike_count",
]
