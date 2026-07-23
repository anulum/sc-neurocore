# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_poisson.py

from __future__ import annotations

"""Full pipeline test for PoissonNeuron (Poisson spike generator).

Stateless process: P(spike in dt) = 1 - exp(-λ·dt/1000).
No membrane dynamics — pure stochastic rate coding."""
from collections.abc import Callable
import math
from typing import cast
import numpy as np
import numpy.typing as npt
import pytest
from sc_neurocore.neurons.models.poisson import PoissonNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count
def _poisson_step_probability(rate_hz: float, dt_ms: float) -> float:
    return -math.expm1(-rate_hz * dt_ms / 1000.0)

__all__ = ['Callable', 'math', 'cast', 'np', 'npt', 'pytest', 'PoissonNeuron', 'Population', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', '_poisson_step_probability']
