# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_pynq_driver.py

from __future__ import annotations

"""Tests for SC-NeuroCore PYNQ FPGA driver."""
import pytest
import numpy as np
import sys
import types
import sc_neurocore.drivers.sc_neurocore_driver as pynq_driver
from sc_neurocore.drivers.physical_twin import PhysicalTwinBridge
from sc_neurocore.drivers.sc_neurocore_driver import SC_NeuroCore_Driver, RealityHardwareError

__all__ = [
    "pytest",
    "np",
    "sys",
    "types",
    "pynq_driver",
    "PhysicalTwinBridge",
    "SC_NeuroCore_Driver",
    "RealityHardwareError",
]
