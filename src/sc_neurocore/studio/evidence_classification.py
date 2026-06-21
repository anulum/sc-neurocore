# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio evidence classification contracts

"""Controlled evidence classes shared by Studio workflow manifests."""

from __future__ import annotations

from typing import Literal, TypeAlias, cast

StudioEvidenceClassification: TypeAlias = Literal[
    "analysis",
    "compile",
    "local_regression",
    "release_benchmark",
    "simulation",
    "synthesis",
    "training",
]
StudioEvidenceStatus: TypeAlias = Literal["completed", "failed", "cancelled", "timed_out"]

STUDIO_EVIDENCE_CLASSIFICATIONS = frozenset(
    {
        "analysis",
        "compile",
        "local_regression",
        "release_benchmark",
        "simulation",
        "synthesis",
        "training",
    }
)
STUDIO_EVIDENCE_TERMINAL_STATUSES = frozenset({"cancelled", "completed", "failed", "timed_out"})


def validate_studio_evidence_classification(value: str) -> StudioEvidenceClassification:
    """Return a controlled Studio evidence class or fail closed.

    Parameters
    ----------
    value:
        Candidate evidence class supplied by a Studio manifest.

    Returns
    -------
    StudioEvidenceClassification
        The validated evidence class.

    Raises
    ------
    ValueError
        If ``value`` is not one of the supported Studio evidence classes.
    """

    if value not in STUDIO_EVIDENCE_CLASSIFICATIONS:
        raise ValueError("Studio evidence classification is invalid.")
    return cast(StudioEvidenceClassification, value)


def validate_studio_evidence_status(value: str) -> StudioEvidenceStatus:
    """Return a controlled terminal evidence status or fail closed.

    Parameters
    ----------
    value:
        Candidate terminal status supplied by a Studio manifest.

    Returns
    -------
    StudioEvidenceStatus
        The validated terminal evidence status.

    Raises
    ------
    ValueError
        If ``value`` is not a supported terminal evidence status.
    """

    if value not in STUDIO_EVIDENCE_TERMINAL_STATUSES:
        raise ValueError("Studio evidence status is invalid.")
    return cast(StudioEvidenceStatus, value)


__all__ = [
    "STUDIO_EVIDENCE_CLASSIFICATIONS",
    "STUDIO_EVIDENCE_TERMINAL_STATUSES",
    "StudioEvidenceClassification",
    "StudioEvidenceStatus",
    "validate_studio_evidence_classification",
    "validate_studio_evidence_status",
]
