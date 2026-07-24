# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_validation_contracts.py

from __future__ import annotations

"""Fail-closed tests for Bioware value and component boundaries."""
from typing import Any, cast
import numpy as np
import pytest
from sc_neurocore.bioware.bioware import (
    AEREvent,
    AERToSCConverter,
    ArtifactRejector,
    BioHybridFrameResult,
    DetectedSpike,
    LatencyBudget,
    LFPBand,
    MEAConfig,
    MEAToAERTranscoder,
    SCToOptoEncoder,
    SpikeDetector,
    SpikeSorter,
    CultureHealth,
    extract_lfp_power,
)
from sc_neurocore.bioware.bioware_validation import (
    require_finite,
    require_nonnegative,
    require_nonnegative_int,
    require_positive,
    require_positive_int,
    validate_binary_bitstream,
    validate_voltage_matrix,
)

__all__ = [
    "Any",
    "cast",
    "np",
    "pytest",
    "AEREvent",
    "AERToSCConverter",
    "ArtifactRejector",
    "BioHybridFrameResult",
    "DetectedSpike",
    "LatencyBudget",
    "LFPBand",
    "MEAConfig",
    "MEAToAERTranscoder",
    "SCToOptoEncoder",
    "SpikeDetector",
    "SpikeSorter",
    "CultureHealth",
    "extract_lfp_power",
    "require_finite",
    "require_nonnegative",
    "require_nonnegative_int",
    "require_positive",
    "require_positive_int",
    "validate_binary_bitstream",
    "validate_voltage_matrix",
]
