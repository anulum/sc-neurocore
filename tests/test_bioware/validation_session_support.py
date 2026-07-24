# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_validation_session.py

from __future__ import annotations

"""Fail-closed and transactional tests for session orchestration."""
from typing import Any, cast
import numpy as np
import pytest
from sc_neurocore.bioware.bioware import (
    AERToSCConverter,
    ArtifactRejector,
    BioHybridSession,
    BiologicalSTDP,
    CultureHealth,
    DetectedSpike,
    HomeostaticPlasticity,
    LatencyBudget,
    MEAConfig,
    MEAToAERTranscoder,
    PharmModel,
    SCToOptoEncoder,
    SpikeDetector,
    SpikeSorter,
)


def _parts() -> dict[str, Any]:
    config = MEAConfig(num_channels=2)
    return {
        "mea_config": config,
        "detector": SpikeDetector(config),
        "transcoder": MEAToAERTranscoder(),
        "sc_converter": AERToSCConverter(num_neurons=2),
        "opto_encoder": SCToOptoEncoder(),
    }


__all__ = [
    "Any",
    "cast",
    "np",
    "pytest",
    "AERToSCConverter",
    "ArtifactRejector",
    "BioHybridSession",
    "BiologicalSTDP",
    "CultureHealth",
    "DetectedSpike",
    "HomeostaticPlasticity",
    "LatencyBudget",
    "MEAConfig",
    "MEAToAERTranscoder",
    "PharmModel",
    "SCToOptoEncoder",
    "SpikeDetector",
    "SpikeSorter",
    "_parts",
]
