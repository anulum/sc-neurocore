# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_sigma_delta.py

from __future__ import annotations

"""Full pipeline test for SigmaDeltaNeuron (Yoon 2017).

Event-driven sigma-delta encoding. Accumulates input in sigma, fires +1
when sigma ≥ θ, fires -1 when sigma ≤ -θ. Subtract-on-spike (not reset).
Ternary output {-1, 0, +1}. Signal reconstruction error bounded by θ."""
import numpy as np
import pytest
from sc_neurocore.neurons.models.sc_sigma_delta_accumulator import (
    SCSigmaDeltaAccumulatorNeuron as SigmaDeltaNeuron,
)
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count

__all__ = [
    "np",
    "pytest",
    "SigmaDeltaNeuron",
    "Population",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "spike_count",
]
