# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_analog_bridge_extended.py

from __future__ import annotations

"""Extended real-surface tests for analog profiles, AER events, and calibration."""
import unittest
from unittest.mock import patch
import numpy as np
import sc_neurocore.analog_bridge as analog_bridge
from sc_neurocore.analog_bridge import (
    AEREvent,
    AnalogBridge,
    AnalogSubstrateProfile,
    CalibrationRoutine,
    EventDrivenInterface,
)

__all__ = [
    "unittest",
    "patch",
    "np",
    "analog_bridge",
    "AEREvent",
    "AnalogBridge",
    "AnalogSubstrateProfile",
    "CalibrationRoutine",
    "EventDrivenInterface",
]
