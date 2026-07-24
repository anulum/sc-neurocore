# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (summary_rejects) from former test_studio_training_evidence.py

from __future__ import annotations

from tests.studio_training_evidence_support import *  # noqa: F403


def test_training_evidence_summary_rejects_unknown_classification() -> None:
    """Training evidence summaries fail closed on unknown evidence classes."""

    summary = build_training_evidence_summary(
        _training_record(),
        _training_payload_reader(evidence_classification="screenshots"),
    )

    assert summary == _unavailable_training_summary()


def test_training_evidence_summary_rejects_non_terminal_status() -> None:
    """Training evidence summaries fail closed on non-terminal evidence statuses."""

    summary = build_training_evidence_summary(
        _training_record(),
        _training_payload_reader(status="running"),
    )

    assert summary == _unavailable_training_summary()


def test_training_evidence_summary_rejects_non_object_payload() -> None:
    """Training evidence summaries fail closed on non-object JSON payloads."""

    summary = build_training_evidence_summary(
        _training_record(),
        _training_raw_payload_reader(["not", "an", "object"]),
    )

    assert summary == _unavailable_training_summary()


def test_training_evidence_summary_rejects_unsupported_schema() -> None:
    """Training evidence summaries fail closed on unsupported schema versions."""

    summary = build_training_evidence_summary(
        _training_record(),
        _training_payload_reader(payload_overrides={"schema_version": "studio.old.v1"}),
    )

    assert summary == _unavailable_training_summary()


def test_training_evidence_summary_rejects_unsupported_action() -> None:
    """Training evidence summaries fail closed on unsupported action kinds."""

    summary = build_training_evidence_summary(
        _training_record(),
        _training_payload_reader(payload_overrides={"action_kind": "studio.compile"}),
    )

    assert summary == _unavailable_training_summary()


def test_training_evidence_summary_rejects_missing_status() -> None:
    """Training evidence summaries fail closed when evidence status is absent."""

    summary = build_training_evidence_summary(
        _training_record(),
        _training_payload_reader(payload_overrides={"status": None}),
    )

    assert summary == _unavailable_training_summary()


def test_training_evidence_summary_rejects_missing_job_id() -> None:
    """Training evidence summaries fail closed when evidence omits job identity."""

    summary = build_training_evidence_summary(
        _training_record(),
        _training_payload_reader(payload_overrides={"job_id": None}),
    )

    assert summary == _unavailable_training_summary()


def test_training_evidence_summary_rejects_empty_required_string() -> None:
    """Training evidence summaries fail closed on empty required string fields."""

    summary = build_training_evidence_summary(
        _training_record(),
        _training_payload_reader(payload_overrides={"payload_sha256": ""}),
    )

    assert summary == _unavailable_training_summary()


def test_training_evidence_summary_rejects_missing_artifact_list() -> None:
    """Training evidence summaries fail closed when result artifact data is absent."""

    summary = build_training_evidence_summary(
        _training_record(),
        _training_payload_reader(payload_overrides={"artifacts": None}),
    )

    assert summary == _unavailable_training_summary()


def test_training_evidence_summary_rejects_non_object_artifact_entry() -> None:
    """Training evidence summaries fail closed on malformed artifact metadata."""

    summary = build_training_evidence_summary(
        _training_record(),
        _training_payload_reader(payload_overrides={"artifacts": ["bad"]}),
    )

    assert summary == _unavailable_training_summary()
