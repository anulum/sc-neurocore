# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_gamma_renewal.py

from __future__ import annotations

"""Hazard-based gamma renewal. Rate-driven. ~83K steps/s."""
import time
import warnings
import numpy as np
import pytest
from sc_neurocore.neurons.models.gamma_renewal import GammaRenewalNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate
class FixedRng:
    def __init__(self, value: float):
        self.value = value

    def random(self) -> float:
        return self.value
def _run(neuron: GammaRenewalNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]

__all__ = ['time', 'warnings', 'np', 'pytest', 'GammaRenewalNeuron', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', 'firing_rate', 'FixedRng', '_run']
