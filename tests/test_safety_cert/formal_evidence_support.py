# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_formal_evidence.py

from __future__ import annotations

"""Focused tests for formal evidence."""
from typing import Any
import pytest
from sc_neurocore.safety_cert.safety_cert import (
    FormalProofCertificate,
    FormalProperty,
    FormalPropertyGapDetector,
    ProofTestCoverage,
    PropertyGap,
    SILLevel,
)


def _unsafe(value: object) -> Any:
    """Return a deliberately invalid runtime value for boundary tests."""
    return value


__all__ = [
    "Any",
    "pytest",
    "FormalProofCertificate",
    "FormalProperty",
    "FormalPropertyGapDetector",
    "ProofTestCoverage",
    "PropertyGap",
    "SILLevel",
    "_unsafe",
]
