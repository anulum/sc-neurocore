# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio policy audit export tests

"""Audit export tests for Studio policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.studio_policy_support import audit_event_hash, policy_contract


def test_jsonl_audit_sink_exports_bounded_recent_events_without_paths(tmp_path: Path) -> None:
    contract = policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    audit_sink = contract["JsonlAuditSink"](audit_path, rotation_bytes=1, retained_files=4)
    for index in range(4):
        audit_sink.record(
            contract["AuditEvent"](
                action=f"studio.simulate.run.{index}",
                route="/api/simulate",
                principal_id="operator-1",
                decision="allow",
                reason="authorized",
            )
        )

    exported = audit_sink.export_recent(limit=3).to_public_dict()

    assert exported["schema_version"] == "studio.audit.export.v1"
    assert exported["sink_type"] == "jsonl"
    assert exported["event_count"] == 3
    assert exported["integrity_error"] is None
    assert exported["integrity_verified"] is True
    assert exported["latest_event_hash"] == exported["events"][-1]["event_hash"]
    assert exported["quarantine_reason"] is None
    assert exported["quarantined_event_count"] == 0
    assert exported["retained_event_count"] == 4
    assert exported["truncated"] is True
    actions = [row["action"] for row in exported["events"]]
    assert actions == [
        "studio.simulate.run.1",
        "studio.simulate.run.2",
        "studio.simulate.run.3",
    ]
    assert str(tmp_path) not in json.dumps(exported)


def test_jsonl_audit_sink_exports_quarantined_legacy_rows(tmp_path: Path) -> None:
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

    exported = audit_sink.export_quarantine().to_public_dict()

    assert exported["schema_version"] == "studio.audit.quarantine.export.v1"
    assert exported["event_count"] == 1
    assert exported["events"][0]["quarantine_reason"] == "legacy_or_unverifiable_rows"
    assert exported["quarantine_reason"] == "legacy_or_unverifiable_rows"
    assert exported["retained_event_count"] == 2
    assert exported["sink_type"] == "jsonl"
    assert exported["truncated"] is False
    assert str(tmp_path) not in json.dumps(exported)


def test_jsonl_audit_sink_exports_chain_break_tail(tmp_path: Path) -> None:
    contract = policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    audit_sink = contract["JsonlAuditSink"](audit_path)
    for index in range(3):
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

    exported = audit_sink.export_quarantine(limit=1).to_public_dict()

    assert exported["event_count"] == 1
    assert exported["events"][0]["action"] == "studio.simulate.run.2"
    assert exported["events"][0]["quarantine_reason"] == "chain_break_rows"
    assert exported["quarantine_reason"] == "chain_break_rows"
    assert exported["retained_event_count"] == 3
    assert exported["truncated"] is True


def test_jsonl_audit_sink_rejects_non_positive_quarantine_export_limit(
    tmp_path: Path,
) -> None:
    contract = policy_contract()
    audit_sink = contract["JsonlAuditSink"](tmp_path / "studio-audit.jsonl")

    with pytest.raises(ValueError, match="positive"):
        audit_sink.export_quarantine(limit=0)


def test_jsonl_audit_sink_exports_empty_quarantine_for_clean_log(tmp_path: Path) -> None:
    """Clean retained audit logs produce an empty quarantine payload."""

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

    exported = audit_sink.export_quarantine().to_public_dict()

    assert exported["event_count"] == 0
    assert exported["events"] == []
    assert exported["quarantine_reason"] is None
    assert exported["retained_event_count"] == 1
    assert exported["truncated"] is False


def test_jsonl_audit_sink_quarantine_export_reports_preflight_failure(
    tmp_path: Path,
) -> None:
    """Quarantine export fails closed when the sink path is malformed."""

    contract = policy_contract()
    audit_sink = contract["JsonlAuditSink"](tmp_path)

    with pytest.raises(contract["AuditSinkError"], match="quarantine export failed"):
        audit_sink.export_quarantine()

    assert audit_sink.status().last_error == "AuditPathIsDirectory"


def test_jsonl_audit_sink_rejects_non_positive_export_limit(tmp_path: Path) -> None:
    contract = policy_contract()
    audit_sink = contract["JsonlAuditSink"](tmp_path / "studio-audit.jsonl")

    with pytest.raises(ValueError, match="positive"):
        audit_sink.export_recent(limit=0)


def test_jsonl_audit_sink_rejects_export_from_directory(tmp_path: Path) -> None:
    contract = policy_contract()
    audit_sink = contract["JsonlAuditSink"](tmp_path)

    with pytest.raises(contract["AuditSinkError"], match="export failed"):
        audit_sink.export_recent()

    assert audit_sink.status().last_error == "AuditPathIsDirectory"


def test_jsonl_audit_sink_rejects_invalid_export_json(tmp_path: Path) -> None:
    contract = policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    audit_path.write_text("{not-json}\n", encoding="utf-8")
    audit_sink = contract["JsonlAuditSink"](audit_path)

    with pytest.raises(contract["AuditSinkError"], match="export failed"):
        audit_sink.export_recent()

    assert audit_sink.status().last_error == "AuditExportInvalidJson"


def test_jsonl_audit_sink_rejects_non_object_export_row(tmp_path: Path) -> None:
    contract = policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    audit_path.write_text('["not", "an", "object"]\n', encoding="utf-8")
    audit_sink = contract["JsonlAuditSink"](audit_path)

    with pytest.raises(contract["AuditSinkError"], match="export failed"):
        audit_sink.export_recent()

    assert audit_sink.status().last_error == "AuditExportInvalidRow"


def test_jsonl_audit_sink_rejects_non_scalar_export_row_value(tmp_path: Path) -> None:
    contract = policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    audit_path.write_text('{"action":["studio.simulate.run"]}\n', encoding="utf-8")
    audit_sink = contract["JsonlAuditSink"](audit_path)

    with pytest.raises(contract["AuditSinkError"], match="export failed"):
        audit_sink.export_recent()

    assert audit_sink.status().last_error == "AuditExportInvalidRow"


def test_jsonl_audit_sink_ignores_blank_export_lines(tmp_path: Path) -> None:
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
    audit_path.write_text(
        f"\n{audit_path.read_text(encoding='utf-8')}\n",
        encoding="utf-8",
    )

    exported = audit_sink.export_recent().to_public_dict()

    assert exported["event_count"] == 1
    assert exported["integrity_verified"] is True


def test_jsonl_audit_sink_sanitizes_export_os_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    original_read_text = Path.read_text

    def blocked_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == audit_path:
            raise PermissionError("blocked path detail")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", blocked_read_text)

    with pytest.raises(contract["AuditSinkError"], match="export failed"):
        audit_sink.export_recent()

    assert audit_sink.status().last_error == "PermissionError"
