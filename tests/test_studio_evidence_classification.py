# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio evidence classification tests

"""Tests for Studio evidence classification contracts."""

from __future__ import annotations

import pytest

from sc_neurocore.studio.evidence_classification import (
    STUDIO_EVIDENCE_CLASSIFICATIONS,
    STUDIO_EVIDENCE_TERMINAL_STATUSES,
    validate_studio_evidence_classification,
    validate_studio_evidence_status,
)
from sc_neurocore.studio.platform import (
    STUDIO_EVIDENCE_CLASSIFICATIONS as PLATFORM_STUDIO_EVIDENCE_CLASSIFICATIONS,
)


def test_studio_evidence_classification_contract_lists_all_manifest_classes() -> None:
    """The shared classification contract covers current Studio evidence surfaces."""

    assert (
        frozenset(
            {
                "analysis",
                "compile",
                "default_flow",
                "local_regression",
                "project_workspace",
                "release_benchmark",
                "simulation",
                "synthesis",
                "training",
            }
        )
        == STUDIO_EVIDENCE_CLASSIFICATIONS
    )
    assert (
        frozenset({"cancelled", "completed", "failed", "timed_out"})
        == STUDIO_EVIDENCE_TERMINAL_STATUSES
    )
    assert PLATFORM_STUDIO_EVIDENCE_CLASSIFICATIONS == STUDIO_EVIDENCE_CLASSIFICATIONS


def test_studio_evidence_classification_validators_return_controlled_values() -> None:
    """Validators return the accepted value for typed manifest construction."""

    assert validate_studio_evidence_classification("analysis") == "analysis"
    assert validate_studio_evidence_classification("compile") == "compile"
    assert validate_studio_evidence_classification("default_flow") == "default_flow"
    assert validate_studio_evidence_classification("project_workspace") == "project_workspace"
    assert validate_studio_evidence_status("completed") == "completed"
    assert validate_studio_evidence_status("timed_out") == "timed_out"


@pytest.mark.parametrize("value", ["", "screenshots", "production_grade", "analysis "])
def test_studio_evidence_classification_validator_rejects_unknown_values(value: str) -> None:
    """Unknown evidence classes fail closed before manifest persistence."""

    with pytest.raises(ValueError, match="classification"):
        validate_studio_evidence_classification(value)


@pytest.mark.parametrize("value", ["", "running", "partial", "completed "])
def test_studio_evidence_status_validator_rejects_non_terminal_values(value: str) -> None:
    """Non-terminal or unknown evidence statuses fail closed."""

    with pytest.raises(ValueError, match="status"):
        validate_studio_evidence_status(value)
