# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio policy audit recording tests

"""Audit recording tests for Studio policy."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.studio_policy_support import audit_event_hash, policy_contract


def test_jsonl_audit_sink_appends_policy_events(tmp_path: Path) -> None:
    contract = policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    audit_sink = contract["JsonlAuditSink"](audit_path)

    audit_sink.record(
        contract["AuditEvent"](
            action="studio.simulate.run",
            route="/api/simulate",
            principal_id="operator-7",
            decision="allow",
            reason="authorized",
        )
    )
    audit_sink.record(
        contract["AuditEvent"](
            action="studio.synth.run",
            route="/api/synth/run",
            principal_id=None,
            decision="deny",
            reason="missing_principal",
        )
    )

    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]

    assert rows[0]["previous_event_hash"] is None
    assert rows[0]["event_hash"] == audit_event_hash(rows[0])
    assert rows[1]["previous_event_hash"] == rows[0]["event_hash"]
    assert rows[1]["event_hash"] == audit_event_hash(rows[1])
    assert rows == [
        rows[0]
        | {
            "action": "studio.simulate.run",
            "decision": "allow",
            "principal_id": "operator-7",
            "reason": "authorized",
            "request_id": None,
            "route": "/api/simulate",
            "schema_version": "studio.audit.v1",
            "timestamp_utc": None,
        },
        rows[1]
        | {
            "action": "studio.synth.run",
            "decision": "deny",
            "principal_id": None,
            "reason": "missing_principal",
            "request_id": None,
            "route": "/api/synth/run",
            "schema_version": "studio.audit.v1",
            "timestamp_utc": None,
        },
    ]


def test_audit_schema_version_is_stable() -> None:
    contract = policy_contract()

    assert contract["AUDIT_SCHEMA_VERSION"] == "studio.audit.v1"
    assert contract["AUDIT_QUARANTINE_EXPORT_SCHEMA_VERSION"] == "studio.audit.quarantine.export.v1"


def test_jsonl_audit_sink_exposes_configured_path(tmp_path: Path) -> None:
    contract = policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    audit_sink = contract["JsonlAuditSink"](audit_path)

    assert audit_sink.path == audit_path


def test_jsonl_audit_sink_rejects_invalid_retention_policy(tmp_path: Path) -> None:
    contract = policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"

    with pytest.raises(ValueError, match="rotation byte"):
        contract["JsonlAuditSink"](audit_path, rotation_bytes=0)
    with pytest.raises(ValueError, match="retained audit"):
        contract["JsonlAuditSink"](audit_path, retained_files=0)


def test_in_memory_audit_sink_reports_non_persistent_status() -> None:
    contract = policy_contract()
    audit_sink = contract["InMemoryAuditSink"]()

    status = audit_sink.status()

    assert status.configured is False
    assert status.healthy is True
    assert status.path_configured is False
    assert status.sink_type == "memory"
    assert status.last_error is None
    assert status.to_public_dict() == {
        "configured": False,
        "healthy": True,
        "integrity_error": None,
        "integrity_verified": None,
        "last_error": None,
        "latest_event_hash": None,
        "path_configured": False,
        "quarantine_reason": None,
        "quarantined_event_count": None,
        "retained_event_count": None,
        "sink_type": "memory",
    }


def test_jsonl_audit_sink_reports_healthy_status(tmp_path: Path) -> None:
    contract = policy_contract()
    audit_sink = contract["JsonlAuditSink"](tmp_path / "studio-audit.jsonl")

    status = audit_sink.status()

    assert status.configured is True
    assert status.healthy is True
    assert status.integrity_error is None
    assert status.integrity_verified is True
    assert status.path_configured is True
    assert status.sink_type == "jsonl"
    assert status.last_error is None
    assert status.latest_event_hash is None
    assert status.quarantine_reason is None
    assert status.quarantined_event_count == 0
    assert status.retained_event_count == 0
