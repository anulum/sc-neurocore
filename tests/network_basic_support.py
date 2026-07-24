# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_network_basic.py

from __future__ import annotations

"""Tests for the declarative network simulation engine."""
from dataclasses import dataclass
import numpy as np
import pytest
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.monitor import SpikeMonitor, StateMonitor, RateMonitor
from sc_neurocore.network.network import Network
from sc_neurocore.network.stimulus import TimedArray, PoissonInput, StepCurrent
from sc_neurocore.network import topology
from sc_neurocore.network.export import export_verilog
from sc_neurocore.exceptions import SCHardwareError

__all__ = [
    "dataclass",
    "np",
    "pytest",
    "Population",
    "Projection",
    "SpikeMonitor",
    "StateMonitor",
    "RateMonitor",
    "Network",
    "TimedArray",
    "PoissonInput",
    "StepCurrent",
    "topology",
    "export_verilog",
    "SCHardwareError",
]
