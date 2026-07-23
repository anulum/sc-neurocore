# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_ccw_bridge.py

from __future__ import annotations

"""Behavioural tests for the SC-NeuroCore ↔ CCW/VIBRANA bridge.

The bridge is a pure data transformation layer (stdlib + numpy, no live CCW
system): it maps SCPN layer metrics onto binaural-audio parameters, L7 glyph
vectors onto VIBRANA visualisation states, and packages both into metadata /
session configs. These tests exercise every mapping, mode-selection branch,
smoothing path, glyph-length guard, and the optional file-export sink.
"""
import json
import numpy as np
import pytest
from sc_neurocore.interfaces.ccw_bridge import (
    CCWBridge,
    CCWMode,
    CCWParameters,
    VIBRANAState,
    create_bridge,
)

__all__ = ['json', 'np', 'pytest', 'CCWBridge', 'CCWMode', 'CCWParameters', 'VIBRANAState', 'create_bridge']
