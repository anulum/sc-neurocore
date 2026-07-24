# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_fractional_lif.py

from __future__ import annotations

"""Full pipeline test for FractionalLIFNeuron (Lundstrom et al. 2008).

Grünwald-Letnikov fractional derivative: D^α v = -(v-v_rest) + R·I.
α<1 introduces memory (power-law decay). History buffer of 100 steps.
GL coefficients: c[k] = c[k-1]·(k-1-α)/k. Performance: ~29K steps/s."""
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.fractional_lif import FractionalLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(neuron: FractionalLIFNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


__all__ = [
    "time",
    "np",
    "pytest",
    "FractionalLIFNeuron",
    "Population",
    "Projection",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "spike_count",
    "firing_rate",
    "_run",
]
