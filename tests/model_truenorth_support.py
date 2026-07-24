# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_truenorth.py

from __future__ import annotations

"""Full pipeline test for TrueNorthNeuron (Merolla 2014, IBM TrueNorth).

Digital integer neuron: v += input - leak. Spike at v ≥ threshold.
Negative saturation at v < -threshold → reset. All-integer arithmetic.
Performance benchmarked: ~1.2M isolation steps/s."""
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.truenorth import TrueNorthNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, isi, firing_rate


def _run(neuron: TrueNorthNeuron, current: int, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


__all__ = [
    "time",
    "np",
    "pytest",
    "TrueNorthNeuron",
    "Population",
    "Projection",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "spike_count",
    "isi",
    "firing_rate",
    "_run",
]
