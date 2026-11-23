# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_learning_advanced.py

"""Shared names and a small two-population network for the advanced learning tests."""

from __future__ import annotations

import numpy as np
from sc_neurocore import StochasticLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.learning.advanced import (
    BPTTLearner,
    TBPTTLearner,
    EligibilityTrace,
    RewardModulatedLearner,
    HomeostaticPlasticity,
    ShortTermPlasticity,
)


def _make_small_network(
    n_in: int = 10, n_out: int = 5, w: float = 0.3, p: float = 0.5
) -> tuple[Network, Projection]:
    pop_in = Population(StochasticLIFNeuron, n=n_in, label="in")
    pop_out = Population(StochasticLIFNeuron, n=n_out, label="out")
    proj = Projection(pop_in, pop_out, weight=w, probability=p, seed=42)
    drive = PoissonInput(n=n_in, rate_hz=100.0, weight=2.0, dt=0.001, seed=99)
    net = Network(pop_in, pop_out, proj, drive)
    return net, proj


__all__ = [
    "np",
    "StochasticLIFNeuron",
    "Population",
    "Projection",
    "Network",
    "PoissonInput",
    "BPTTLearner",
    "TBPTTLearner",
    "EligibilityTrace",
    "RewardModulatedLearner",
    "HomeostaticPlasticity",
    "ShortTermPlasticity",
    "_make_small_network",
]
