# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_brainscales_adex.py

from __future__ import annotations

"""Full pipeline test for BrainScaleSAdExNeuron (Schemmel 2010).

BrainScaleS-2 analog AdEx with 1000× hardware speedup emulation.
Clipped exponential for numerical safety."""
import numpy as np
import pytest
from sc_neurocore.neurons.models.brainscales_adex import BrainScaleSAdExNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count, isi

__all__ = [
    "np",
    "pytest",
    "BrainScaleSAdExNeuron",
    "Population",
    "Projection",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "firing_rate",
    "spike_count",
    "isi",
]
