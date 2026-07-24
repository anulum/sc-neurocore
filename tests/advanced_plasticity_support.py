# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_advanced_plasticity.py

from __future__ import annotations

"""Tests for advanced plasticity: BPTT, eligibility, R-STDP, meta, homeostatic, STP, structural."""
import numpy as np
import pytest
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.learning.advanced import (
    BPTTLearner,
    TBPTTLearner,
    EligibilityTrace,
    HomeostaticPlasticity,
    MetaLearner,
    RewardModulatedLearner,
    ShortTermPlasticity,
    StructuralPlasticity,
    _fast_sigmoid_surrogate,
)


@pytest.fixture()
def simple_net():
    """Two-population network with one projection."""
    pop_a = Population("LapicqueNeuron", 5, label="src")
    pop_b = Population("LapicqueNeuron", 5, label="tgt")
    proj = Projection(pop_a, pop_b, weight=0.3, probability=1.0, seed=0)
    net = Network(pop_a, pop_b, proj)
    return net, pop_a, pop_b, proj


__all__ = [
    "np",
    "pytest",
    "Population",
    "Projection",
    "Network",
    "BPTTLearner",
    "TBPTTLearner",
    "EligibilityTrace",
    "HomeostaticPlasticity",
    "MetaLearner",
    "RewardModulatedLearner",
    "ShortTermPlasticity",
    "StructuralPlasticity",
    "_fast_sigmoid_surrogate",
    "simple_net",
]
