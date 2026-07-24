# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_validation_experiment.py

from __future__ import annotations

"""Negative-path tests for experiment, audit, plasticity, and fitness APIs."""
from typing import Any, cast
import numpy as np
import pytest
from sc_neurocore.bioware.bioware import (
    BCMPlasticity,
    BioAuditEntry,
    BioAuditLog,
    BiologicalSTDP,
    DetectedSpike,
    HomeostaticPlasticity,
    MEAConfig,
    MultiWellPlate,
    PharmModel,
    WellConfig,
    _mea_response_latency_ms,
    _quantile_indices,
    mea_fitness_hook,
)


def _entry(round_number: int = 1) -> BioAuditEntry:
    return BioAuditEntry(round_number, "2026-07-13T08:00:00+00:00", 1, 0, 1.0, 0.9)


__all__ = [
    "Any",
    "cast",
    "np",
    "pytest",
    "BCMPlasticity",
    "BioAuditEntry",
    "BioAuditLog",
    "BiologicalSTDP",
    "DetectedSpike",
    "HomeostaticPlasticity",
    "MEAConfig",
    "MultiWellPlate",
    "PharmModel",
    "WellConfig",
    "_mea_response_latency_ms",
    "_quantile_indices",
    "mea_fitness_hook",
    "_entry",
]
