# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_escape_rate.py

from __future__ import annotations

"""Full pipeline tests for the maintained Gerstner-2000 EscapeRate cell.

The exact constant-current RC candidate drives the finite-step probability
``1 - exp(-rho_0 * exp((v-v_threshold)/delta_u) * dt)``. One private seeded
LFSR16 trial then commits either ``v_reset`` or that membrane candidate.
"""
import time
import math
from typing import cast
import numpy as np
import pytest
from sc_neurocore.neurons._stochastic_threshold import (
    DEFAULT_LFSR16_SEED,
    LFSR16_ADVANCES_PER_TRIAL,
    Lfsr16Threshold,
    lfsr16_advance,
    lfsr16_trial_sample,
    probability_to_lfsr16_threshold,
)
from sc_neurocore.neurons.models.escape_rate import EscapeRateNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, isi, firing_rate


def _run(neuron: EscapeRateNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


__all__ = [
    "time",
    "math",
    "cast",
    "np",
    "pytest",
    "DEFAULT_LFSR16_SEED",
    "LFSR16_ADVANCES_PER_TRIAL",
    "Lfsr16Threshold",
    "lfsr16_advance",
    "lfsr16_trial_sample",
    "probability_to_lfsr16_threshold",
    "EscapeRateNeuron",
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
