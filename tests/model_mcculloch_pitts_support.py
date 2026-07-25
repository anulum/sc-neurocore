# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — McCulloch-Pitts test support

"""Shared imports and constants for McCulloch-Pitts model tests."""

from __future__ import annotations

import dataclasses
from typing import cast

import numpy as np
import pytest

from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.network import Network
from sc_neurocore.network.population import Population
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.neurons.models.mcculloch_pitts import (
    McCullochPittsNeuron,
    encode_hardware_input,
)

_INT32_MAX = (1 << 31) - 1

__all__ = [
    "McCullochPittsNeuron",
    "Network",
    "PoissonInput",
    "Population",
    "SpikeMonitor",
    "_INT32_MAX",
    "cast",
    "dataclasses",
    "encode_hardware_input",
    "firing_rate",
    "np",
    "pytest",
    "spike_count",
]
