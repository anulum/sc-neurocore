# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_fault_injection.py

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
import numpy as np
import pytest
from sc_neurocore.fault_injection.fault_injection import (
    FaultInjector,
    FaultModel,
    FaultInjectionResult,
    RadiationProfile,
    ResilienceBenchmark,
    ResilienceReport,
)
_VALID_REPORT = dict(
    fault_model="bit_flip",
    ber=0.1,
    bitstream_length=10,
    num_trials=5,
    mean_error=0.1,
    std_error=0.05,
    max_error=0.3,
    p95_error=0.2,
    p99_error=0.25,
    mean_bits_flipped=1.0,
    wall_time_ms=1.0,
)

__all__ = ['unittest', 'MagicMock', 'np', 'pytest', 'FaultInjector', 'FaultModel', 'FaultInjectionResult', 'RadiationProfile', 'ResilienceBenchmark', 'ResilienceReport', '_VALID_REPORT']
