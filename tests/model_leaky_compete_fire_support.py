# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_leaky_compete_fire.py

from __future__ import annotations

"""Module-specific numerical and pipeline tests for LeakyCompeteFireNeuron.

Winner-take-all with lateral inhibition. Multi-unit model (n_units=4):
tau dV_i/dt = -V_i + I_i, integrated by exact first-order relaxation.
On spike: V_i -> 0, V_j -= w_inh (j != i), clipped >= 0."""
import math
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.leaky_compete_fire import LeakyCompeteFireNeuron
from sc_neurocore.network.population import Population


def _exact_lcf_candidates(neuron: LeakyCompeteFireNeuron, currents: list[float]) -> list[float]:
    decay = math.exp(-neuron.dt / neuron.tau)
    return [current + (voltage - current) * decay for voltage, current in zip(neuron.v, currents)]


__all__ = [
    "math",
    "time",
    "np",
    "pytest",
    "LeakyCompeteFireNeuron",
    "Population",
    "_exact_lcf_candidates",
]
