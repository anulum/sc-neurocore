# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_akida_neuron.py

from __future__ import annotations

"""Full pipeline test for AkidaNeuron (BrainChip Akida 2021).

Event-domain rank-order integrate-and-fire neuron:
V += int(weight · modulation^rank)
rank increments per non-zero input event.

Key properties:
- Integer arithmetic (V: int, weight: int)
- Rank-order coding: earlier events weighted more (modulation=0.75)
- Single-spike model: fires AT MOST ONCE (_spiked flag)
- No leak between events
- No reset after spike (just flags _spiked)

Performance: ~1.1M steps/s (integer arithmetic).
FULL PIPELINE WIRED + PERFORMANCE."""
import time
import os
import numpy as np
import pytest
from sc_neurocore.neurons.models.akida_neuron import AkidaNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count

__all__ = ['time', 'os', 'np', 'pytest', 'AkidaNeuron', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count']
