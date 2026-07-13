# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio policy audit integrity tests

"""Audit integrity tests for Studio policy."""

from __future__ import annotations

import json
from pathlib import Path

from tests.studio_policy_support import audit_event_hash, policy_contract


def test_jsonl_audit_sink_status_reports_hash_mismatch_without_paths(
    tmp_path: Path,
) -> None:
    """Audit status reports retained-row tampering without exposing paths."""

    contract = policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    audit_sink = contract["JsonlAuditSink"](audit_path)
    audit_sink.record(
        contract["AuditEvent"](
            action="studio.simulate.run",
            route="/api/simulate",
            principal_id="operator-1",
            decision="allow",
            reason="authorized",
        )
    )
    row = json.loads(audit_path.read_text(encoding="utf-8"))
    row["reason"] = "tampered"
    audit_path.write_text(json.dumps(row, sort_keys=True), encoding="utf-8")

    status = audit_sink.status().to_public_dict()
    exported = audit_sink.export_recent().to_public_dict()
    quarantine_export = audit_sink.export_quarantine().to_public_dict()

    assert status["healthy"] is False
    assert status["integrity_verified"] is False
    assert status["integrity_error"] == "AuditIntegrityHashMismatch"
    assert status["last_error"] == "AuditIntegrityHashMismatch"
    assert status["quarantine_reason"] == "tampered_or_corrupt_rows"
    assert status["quarantined_event_count"] == 1
    assert status["retained_event_count"] == 1
    assert str(tmp_path) not in json.dumps(status)
    assert exported["integrity_verified"] is False
    assert exported["integrity_error"] == "AuditIntegrityHashMismatch"
    assert exported["quarantine_reason"] == "tampered_or_corrupt_rows"
    assert exported["quarantined_event_count"] == 1
    assert exported["retained_event_count"] == 1
    assert quarantine_export["event_count"] == 1
    assert quarantine_export["quarantine_reason"] == "tampered_or_corrupt_rows"
    assert quarantine_export["events"][0]["quarantine_reason"] == "tampered_or_corrupt_rows"
    assert str(tmp_path) not in json.dumps(exported)
    assert str(tmp_path) not in json.dumps(quarantine_export)


def test_jsonl_audit_sink_quarantine_export_summarizes_mixed_reasons(
    tmp_path: Path,
) -> None:
    """Quarantine export preserves rows when multiple defect classes exist."""

    contract = policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    audit_path.write_text('{"schema_version":"studio.audit.v1"}\n', encoding="utf-8")
    audit_sink = contract["JsonlAuditSink"](audit_path)
    for index in range(2):
        audit_sink.record(
            contract["AuditEvent"](
                action=f"studio.simulate.run.{index}",
                route="/api/simulate",
                principal_id="operator-1",
                decision="allow",
                reason="authorized",
            )
        )
    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    rows[2]["reason"] = "tampered"
    audit_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows),
        encoding="utf-8",
    )

    exported = audit_sink.export_quarantine().to_public_dict()

    assert exported["event_count"] == 2
    assert exported["quarantine_reason"] == "multiple_quarantine_reasons"
    assert [event["quarantine_reason"] for event in exported["events"]] == [
        "legacy_or_unverifiable_rows",
        "tampered_or_corrupt_rows",
    ]
    assert str(tmp_path) not in json.dumps(exported)


def test_jsonl_audit_sink_quarantines_legacy_rows_without_paths(tmp_path: Path) -> None:
    """Legacy rows remain exportable but are counted as quarantined."""

    contract = policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    audit_path.write_text('{"schema_version":"studio.audit.v1"}\n', encoding="utf-8")
    audit_sink = contract["JsonlAuditSink"](audit_path)
    audit_sink.record(
        contract["AuditEvent"](
            action="studio.simulate.run",
            route="/api/simulate",
            principal_id="operator-1",
            decision="allow",
            reason="authorized",
        )
    )

    status = audit_sink.status().to_public_dict()
    exported = audit_sink.export_recent().to_public_dict()

    assert status["healthy"] is False
    assert status["integrity_error"] == "AuditIntegrityMissingHash"
    assert status["integrity_verified"] is False
    assert status["latest_event_hash"] == exported["events"][-1]["event_hash"]
    assert status["quarantine_reason"] == "legacy_or_unverifiable_rows"
    assert status["quarantined_event_count"] == 1
    assert status["retained_event_count"] == 2
    assert exported["quarantine_reason"] == "legacy_or_unverifiable_rows"
    assert exported["quarantined_event_count"] == 1
    assert exported["retained_event_count"] == 2
    assert str(tmp_path) not in json.dumps(status)
    assert str(tmp_path) not in json.dumps(exported)


def test_jsonl_audit_sink_status_reports_chain_mismatch(tmp_path: Path) -> None:
    """Audit status rejects retained rows with broken previous-hash links."""

    contract = policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    audit_sink = contract["JsonlAuditSink"](audit_path)
    for index in range(2):
        audit_sink.record(
            contract["AuditEvent"](
                action=f"studio.simulate.run.{index}",
                route="/api/simulate",
                principal_id="operator-1",
                decision="allow",
                reason="authorized",
            )
        )
    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    rows[1]["previous_event_hash"] = "0" * 64
    rows[1]["event_hash"] = audit_event_hash(rows[1])
    audit_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows),
        encoding="utf-8",
    )

    status = audit_sink.status()

    assert status.healthy is False
    assert status.integrity_verified is False
    assert status.integrity_error == "AuditIntegrityChainMismatch"
    assert status.last_error == "AuditIntegrityChainMismatch"
    assert status.quarantine_reason == "chain_break_rows"
    assert status.quarantined_event_count == 1
    assert status.retained_event_count == 2
