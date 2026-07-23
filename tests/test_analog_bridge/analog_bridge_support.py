# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_analog_bridge.py

from __future__ import annotations

"""Real-surface tests for analog substrate DAC configuration emission."""
import unittest
from dataclasses import dataclass
from sc_neurocore.analog_bridge import AnalogBridge
@dataclass
class MockNode:
    """Node descriptor matching the bridge's public configuration contract."""

    type: str
    id: str
    probability: float = 0.0
    threshold: float = 0.0

__all__ = ['unittest', 'dataclass', 'AnalogBridge', 'MockNode']
