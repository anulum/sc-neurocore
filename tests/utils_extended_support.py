# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_utils_extended.py

from __future__ import annotations

"""
Tests for the 6 untested core utils modules:
  - adaptive.py (AdaptiveInference)
  - connectomes.py (ConnectomeGenerator)
  - decorrelators.py (ShufflingDecorrelator, LFSRRegenDecorrelator)
  - fault_injection.py (FaultInjector)
  - fsm_activations.py (TanhFSM, ReLKFSM)
  - model_bridge.py (normalize_weights, SCBridge)
"""
import pytest
import numpy as np
from sc_neurocore.utils.adaptive import AdaptiveInference
from sc_neurocore.utils.connectomes import ConnectomeGenerator
from sc_neurocore.utils.decorrelators import (
    ShufflingDecorrelator,
    LFSRRegenDecorrelator,
)
from sc_neurocore.utils.fault_injection import FaultInjector
from sc_neurocore.utils.fsm_activations import TanhFSM, ReLKFSM
from sc_neurocore.utils.model_bridge import normalize_weights, SCBridge

__all__ = [
    "pytest",
    "np",
    "AdaptiveInference",
    "ConnectomeGenerator",
    "ShufflingDecorrelator",
    "LFSRRegenDecorrelator",
    "FaultInjector",
    "TanhFSM",
    "ReLKFSM",
    "normalize_weights",
    "SCBridge",
]
