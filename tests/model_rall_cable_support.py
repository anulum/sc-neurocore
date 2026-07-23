# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_rall_cable.py

from __future__ import annotations

"""Full pipeline test for RallCableNeuron (Rall 1962).

N-compartment passive cable. Current injected at distal end (N-1),
spike detected at soma (compartment 0). Signal attenuates with distance."""
import numpy as np
import pytest
from typing import Any
from sc_neurocore.neurons.models.rall_cable import RallCableNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi
def _run(neuron: RallCableNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]

__all__ = ['np', 'pytest', 'Any', 'RallCableNeuron', 'Population', 'spike_count', 'firing_rate', 'isi', '_run']
