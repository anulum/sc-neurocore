# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio policy gateway contract tests

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

UTC = timezone.utc


def _policy_contract() -> dict[str, Any]:
    try:
        from sc_neurocore.studio.platform.policy import (  # noqa: PLC0415
            AuditEvent,
            AUDIT_SCHEMA_VERSION,
            AUDIT_QUARANTINE_EXPORT_SCHEMA_VERSION,
            AuditSinkError,
            AuditSinkStatus,
            InMemoryAuditSink,
            JsonlAuditSink,
            PolicyGateway,
            Principal,
            RoutePolicyRegistry,
            RoutePolicy,
            RouteVisibility,
            build_default_studio_route_policy_registry,
        )
    except ImportError as exc:
        pytest.fail(f"Studio policy contract is missing: {exc}")
    return {
        "AuditEvent": AuditEvent,
        "AUDIT_SCHEMA_VERSION": AUDIT_SCHEMA_VERSION,
        "AUDIT_QUARANTINE_EXPORT_SCHEMA_VERSION": AUDIT_QUARANTINE_EXPORT_SCHEMA_VERSION,
        "AuditSinkError": AuditSinkError,
        "AuditSinkStatus": AuditSinkStatus,
        "InMemoryAuditSink": InMemoryAuditSink,
        "JsonlAuditSink": JsonlAuditSink,
        "PolicyGateway": PolicyGateway,
        "Principal": Principal,
        "RoutePolicyRegistry": RoutePolicyRegistry,
        "RoutePolicy": RoutePolicy,
        "RouteVisibility": RouteVisibility,
        "build_default_studio_route_policy_registry": build_default_studio_route_policy_registry,
    }


def _audit_event_hash(row: dict[str, Any]) -> str:
    unsigned_row = dict(row)
    unsigned_row.pop("event_hash", None)
    canonical_row = json.dumps(
        unsigned_row,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical_row).hexdigest()


def test_policy_gateway_allows_public_route_without_principal() -> None:
    contract = _policy_contract()
    gateway = contract["PolicyGateway"](audit_sink=contract["InMemoryAuditSink"]())
    policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].PUBLIC,
        audit_action="studio.health.read",
    )

    decision = gateway.authorize(policy, principal=None, route="/api/health")

    assert decision.allowed is True
    assert decision.reason == "public_route"
    assert decision.status_code == 200


def test_policy_gateway_requires_principal_for_authenticated_route() -> None:
    contract = _policy_contract()
    audit_sink = contract["InMemoryAuditSink"]()
    gateway = contract["PolicyGateway"](audit_sink=audit_sink)
    policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].AUTHENTICATED,
        audit_action="studio.capabilities.read",
    )

    decision = gateway.authorize(policy, principal=None, route="/api/studio/capabilities")

    assert decision.allowed is False
    assert decision.reason == "missing_principal"
    assert decision.status_code == 401
    assert audit_sink.events[-1].decision == "deny"
    assert audit_sink.events[-1].principal_id is None


def test_policy_gateway_denies_missing_role_and_records_audit_event() -> None:
    contract = _policy_contract()
    audit_sink = contract["InMemoryAuditSink"]()
    gateway = contract["PolicyGateway"](audit_sink=audit_sink)
    policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].AUTHENTICATED,
        required_roles=frozenset({"studio.admin"}),
        audit_action="studio.policy.write",
    )
    principal = contract["Principal"](principal_id="operator-1", roles=frozenset({"studio.viewer"}))

    decision = gateway.authorize(policy, principal=principal, route="/api/studio/policy")

    assert decision.allowed is False
    assert decision.reason == "missing_required_role"
    assert decision.status_code == 403
    assert audit_sink.events[-1].action == "studio.policy.write"
    assert audit_sink.events[-1].route == "/api/studio/policy"
    assert audit_sink.events[-1].principal_id == "operator-1"
    assert audit_sink.events[-1].decision == "deny"


def test_policy_gateway_allows_authenticated_route_with_required_role() -> None:
    contract = _policy_contract()
    audit_sink = contract["InMemoryAuditSink"]()
    gateway = contract["PolicyGateway"](audit_sink=audit_sink)
    policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].AUTHENTICATED,
        required_roles=frozenset({"studio.viewer"}),
        audit_action="studio.capabilities.read",
    )
    principal = contract["Principal"](principal_id="operator-3", roles=frozenset({"studio.viewer"}))

    decision = gateway.authorize(policy, principal=principal, route="/api/studio/capabilities")

    assert decision.allowed is True
    assert decision.reason == "authorized"
    assert decision.status_code == 200
    assert audit_sink.events[-1].decision == "allow"
    assert audit_sink.events[-1].principal_id == "operator-3"


def test_jsonl_audit_sink_appends_policy_events(tmp_path: Path) -> None:
    contract = _policy_contract()
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
    assert rows[0]["event_hash"] == _audit_event_hash(rows[0])
    assert rows[1]["previous_event_hash"] == rows[0]["event_hash"]
    assert rows[1]["event_hash"] == _audit_event_hash(rows[1])
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
    contract = _policy_contract()

    assert contract["AUDIT_SCHEMA_VERSION"] == "studio.audit.v1"
    assert (
        contract["AUDIT_QUARANTINE_EXPORT_SCHEMA_VERSION"]
        == "studio.audit.quarantine.export.v1"
    )


def test_jsonl_audit_sink_exposes_configured_path(tmp_path: Path) -> None:
    contract = _policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    audit_sink = contract["JsonlAuditSink"](audit_path)

    assert audit_sink.path == audit_path


def test_jsonl_audit_sink_rejects_invalid_retention_policy(tmp_path: Path) -> None:
    contract = _policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"

    with pytest.raises(ValueError, match="rotation byte"):
        contract["JsonlAuditSink"](audit_path, rotation_bytes=0)
    with pytest.raises(ValueError, match="retained audit"):
        contract["JsonlAuditSink"](audit_path, retained_files=0)


def test_in_memory_audit_sink_reports_non_persistent_status() -> None:
    contract = _policy_contract()
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
    contract = _policy_contract()
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


def test_jsonl_audit_sink_reports_failed_append_policy(tmp_path: Path) -> None:
    contract = _policy_contract()
    audit_sink = contract["JsonlAuditSink"](tmp_path)

    with pytest.raises(contract["AuditSinkError"], match="append failed"):
        audit_sink.record(
            contract["AuditEvent"](
                action="studio.simulate.run",
                route="/api/simulate",
                principal_id="operator-7",
                decision="allow",
                reason="authorized",
            )
        )

    status = audit_sink.status()
    assert status.configured is True
    assert status.healthy is False
    assert status.path_configured is True
    assert status.sink_type == "jsonl"
    assert status.last_error == "AuditPathIsDirectory"


def test_jsonl_audit_sink_status_rejects_directory_log_path(tmp_path: Path) -> None:
    contract = _policy_contract()
    audit_sink = contract["JsonlAuditSink"](tmp_path)

    status = audit_sink.status()

    assert status.configured is True
    assert status.healthy is False
    assert status.last_error == "AuditPathIsDirectory"


def test_jsonl_audit_sink_status_rejects_file_parent(tmp_path: Path) -> None:
    contract = _policy_contract()
    audit_parent = tmp_path / "not-a-directory"
    audit_parent.write_text("not a directory", encoding="utf-8")
    audit_sink = contract["JsonlAuditSink"](audit_parent / "studio.jsonl")

    status = audit_sink.status()

    assert status.configured is True
    assert status.healthy is False
    assert status.last_error == "AuditParentIsNotDirectory"


def test_jsonl_audit_sink_sanitizes_append_os_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    audit_sink = contract["JsonlAuditSink"](audit_path)
    original_open = Path.open

    def blocked_open(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == audit_path and args and args[0] == "a":
            raise PermissionError("blocked path detail")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", blocked_open)

    with pytest.raises(contract["AuditSinkError"], match="append failed"):
        audit_sink.record(
            contract["AuditEvent"](
                action="studio.simulate.run",
                route="/api/simulate",
                principal_id="operator-7",
                decision="allow",
                reason="authorized",
            )
        )

    status = audit_sink.status()
    assert status.healthy is False
    assert status.last_error == "PermissionError"


def test_jsonl_audit_sink_rotates_and_retains_hash_chain(tmp_path: Path) -> None:
    contract = _policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    audit_sink = contract["JsonlAuditSink"](
        audit_path,
        rotation_bytes=1,
        retained_files=2,
    )

    for index in range(4):
        audit_sink.record(
            contract["AuditEvent"](
                action=f"studio.simulate.run.{index}",
                route="/api/simulate",
                principal_id="operator-7",
                decision="allow",
                reason="authorized",
            )
        )

    current_row = json.loads(audit_path.read_text(encoding="utf-8"))
    rotated_latest = json.loads(
        audit_path.with_name("studio-audit.jsonl.1").read_text(encoding="utf-8")
    )
    rotated_retained = json.loads(
        audit_path.with_name("studio-audit.jsonl.2").read_text(encoding="utf-8")
    )

    assert current_row["action"] == "studio.simulate.run.3"
    assert rotated_latest["action"] == "studio.simulate.run.2"
    assert rotated_retained["action"] == "studio.simulate.run.1"
    assert not audit_path.with_name("studio-audit.jsonl.3").exists()
    assert current_row["previous_event_hash"] == rotated_latest["event_hash"]
    assert rotated_latest["previous_event_hash"] == rotated_retained["event_hash"]
    assert current_row["event_hash"] == _audit_event_hash(current_row)
    assert audit_sink.status().integrity_verified is True


def test_jsonl_audit_sink_exports_bounded_recent_events_without_paths(tmp_path: Path) -> None:
    contract = _policy_contract()
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
    contract = _policy_contract()
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
    contract = _policy_contract()
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
    rows[1]["event_hash"] = _audit_event_hash(rows[1])
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
    contract = _policy_contract()
    audit_sink = contract["JsonlAuditSink"](tmp_path / "studio-audit.jsonl")

    with pytest.raises(ValueError, match="positive"):
        audit_sink.export_quarantine(limit=0)


def test_jsonl_audit_sink_exports_empty_quarantine_for_clean_log(tmp_path: Path) -> None:
    """Clean retained audit logs produce an empty quarantine payload."""

    contract = _policy_contract()
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

    contract = _policy_contract()
    audit_sink = contract["JsonlAuditSink"](tmp_path)

    with pytest.raises(contract["AuditSinkError"], match="quarantine export failed"):
        audit_sink.export_quarantine()

    assert audit_sink.status().last_error == "AuditPathIsDirectory"


def test_jsonl_audit_sink_status_reports_hash_mismatch_without_paths(
    tmp_path: Path,
) -> None:
    """Audit status reports retained-row tampering without exposing paths."""

    contract = _policy_contract()
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

    contract = _policy_contract()
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
    assert [
        event["quarantine_reason"] for event in exported["events"]
    ] == [
        "legacy_or_unverifiable_rows",
        "tampered_or_corrupt_rows",
    ]
    assert str(tmp_path) not in json.dumps(exported)


def test_jsonl_audit_sink_quarantines_legacy_rows_without_paths(tmp_path: Path) -> None:
    """Legacy rows remain exportable but are counted as quarantined."""

    contract = _policy_contract()
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

    contract = _policy_contract()
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
    rows[1]["event_hash"] = _audit_event_hash(rows[1])
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


def test_jsonl_audit_sink_rejects_non_positive_export_limit(tmp_path: Path) -> None:
    contract = _policy_contract()
    audit_sink = contract["JsonlAuditSink"](tmp_path / "studio-audit.jsonl")

    with pytest.raises(ValueError, match="positive"):
        audit_sink.export_recent(limit=0)


def test_jsonl_audit_sink_rejects_export_from_directory(tmp_path: Path) -> None:
    contract = _policy_contract()
    audit_sink = contract["JsonlAuditSink"](tmp_path)

    with pytest.raises(contract["AuditSinkError"], match="export failed"):
        audit_sink.export_recent()

    assert audit_sink.status().last_error == "AuditPathIsDirectory"


def test_jsonl_audit_sink_rejects_invalid_export_json(tmp_path: Path) -> None:
    contract = _policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    audit_path.write_text("{not-json}\n", encoding="utf-8")
    audit_sink = contract["JsonlAuditSink"](audit_path)

    with pytest.raises(contract["AuditSinkError"], match="export failed"):
        audit_sink.export_recent()

    assert audit_sink.status().last_error == "AuditExportInvalidJson"


def test_jsonl_audit_sink_rejects_non_object_export_row(tmp_path: Path) -> None:
    contract = _policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    audit_path.write_text('["not", "an", "object"]\n', encoding="utf-8")
    audit_sink = contract["JsonlAuditSink"](audit_path)

    with pytest.raises(contract["AuditSinkError"], match="export failed"):
        audit_sink.export_recent()

    assert audit_sink.status().last_error == "AuditExportInvalidRow"


def test_jsonl_audit_sink_rejects_non_scalar_export_row_value(tmp_path: Path) -> None:
    contract = _policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    audit_path.write_text('{"action":["studio.simulate.run"]}\n', encoding="utf-8")
    audit_sink = contract["JsonlAuditSink"](audit_path)

    with pytest.raises(contract["AuditSinkError"], match="export failed"):
        audit_sink.export_recent()

    assert audit_sink.status().last_error == "AuditExportInvalidRow"


def test_policy_gateway_accepts_jsonl_audit_sink(tmp_path: Path) -> None:
    contract = _policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    gateway = contract["PolicyGateway"](audit_sink=contract["JsonlAuditSink"](audit_path))
    policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].AUTHENTICATED,
        audit_action="studio.simulate.run",
    )

    decision = gateway.authorize(policy, principal=None, route="/api/simulate")
    row = json.loads(audit_path.read_text(encoding="utf-8"))

    assert decision.allowed is False
    assert row["decision"] == "deny"
    assert row["reason"] == "missing_principal"
    assert row["schema_version"] == "studio.audit.v1"


def test_jsonl_audit_sink_starts_chain_after_blank_log(tmp_path: Path) -> None:
    contract = _policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    audit_path.write_text("\n\n", encoding="utf-8")
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

    row = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[-1])

    assert row["previous_event_hash"] is None
    assert row["event_hash"] == _audit_event_hash(row)


def test_jsonl_audit_sink_ignores_blank_export_lines(tmp_path: Path) -> None:
    contract = _policy_contract()
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


def test_jsonl_audit_sink_starts_chain_after_legacy_row(tmp_path: Path) -> None:
    contract = _policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    audit_path.write_text('{"schema_version":"studio.audit.v1"}\n', encoding="utf-8")
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

    row = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[-1])

    assert row["previous_event_hash"] is None
    assert row["event_hash"] == _audit_event_hash(row)


def test_jsonl_audit_sink_sanitizes_export_os_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _policy_contract()
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


def test_policy_gateway_records_request_id_in_audit_event(tmp_path: Path) -> None:
    contract = _policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    gateway = contract["PolicyGateway"](audit_sink=contract["JsonlAuditSink"](audit_path))
    policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].AUTHENTICATED,
        audit_action="studio.simulate.run",
    )

    decision = gateway.authorize(
        policy,
        principal=None,
        route="/api/simulate",
        request_id="studio-run-42",
    )
    row = json.loads(audit_path.read_text(encoding="utf-8"))

    assert decision.allowed is False
    assert row["request_id"] == "studio-run-42"


def test_policy_gateway_records_injected_utc_timestamp(tmp_path: Path) -> None:
    contract = _policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    timestamp = datetime(2026, 6, 19, 3, 52, 0, tzinfo=UTC)
    gateway = contract["PolicyGateway"](
        audit_sink=contract["JsonlAuditSink"](audit_path),
        clock=lambda: timestamp,
    )
    policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].AUTHENTICATED,
        audit_action="studio.simulate.run",
    )

    decision = gateway.authorize(policy, principal=None, route="/api/simulate")
    row = json.loads(audit_path.read_text(encoding="utf-8"))

    assert decision.allowed is False
    assert row["timestamp_utc"] == "2026-06-19T03:52:00Z"


def test_policy_gateway_records_default_utc_timestamp(tmp_path: Path) -> None:
    contract = _policy_contract()
    audit_path = tmp_path / "studio-audit.jsonl"
    gateway = contract["PolicyGateway"](audit_sink=contract["JsonlAuditSink"](audit_path))
    policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].AUTHENTICATED,
        audit_action="studio.simulate.run",
    )

    decision = gateway.authorize(policy, principal=None, route="/api/simulate")
    row = json.loads(audit_path.read_text(encoding="utf-8"))

    assert decision.allowed is False
    assert row["timestamp_utc"].endswith("Z")
    assert datetime.fromisoformat(row["timestamp_utc"].replace("Z", "+00:00")).tzinfo is UTC


def test_policy_gateway_rejects_admin_route_without_admin_role() -> None:
    contract = _policy_contract()
    audit_sink = contract["InMemoryAuditSink"]()
    gateway = contract["PolicyGateway"](audit_sink=audit_sink)
    policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].ADMIN,
        audit_action="studio.admin.configure",
    )
    principal = contract["Principal"](principal_id="operator-2", roles=frozenset({"studio.viewer"}))

    decision = gateway.authorize(policy, principal=principal, route="/api/studio/admin")

    assert decision.allowed is False
    assert decision.reason == "missing_admin_role"
    assert decision.status_code == 403


def test_route_policy_rejects_empty_audit_action_for_protected_route() -> None:
    contract = _policy_contract()

    with pytest.raises(ValueError, match="audit_action"):
        contract["RoutePolicy"](visibility=contract["RouteVisibility"].AUTHENTICATED)


def test_route_policy_registry_rejects_duplicate_method_path() -> None:
    contract = _policy_contract()
    registry = contract["RoutePolicyRegistry"]()
    policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].PUBLIC,
        audit_action="studio.health.read",
    )
    registry.register("GET", "/api/health", policy)

    with pytest.raises(ValueError, match="already has a Studio route policy"):
        registry.register("get", "/api/health", policy)


def test_default_route_policy_registry_classifies_platform_routes() -> None:
    contract = _policy_contract()
    registry = contract["build_default_studio_route_policy_registry"]()

    health_policy = registry.policy_for("GET", "/api/health")
    capability_policy = registry.policy_for("GET", "/api/studio/capabilities")
    detail_policy = registry.policy_for("GET", "/api/studio/capabilities/{capability_id}")
    jobs_list_policy = registry.policy_for("GET", "/api/studio/jobs")
    job_detail_policy = registry.policy_for("GET", "/api/studio/jobs/{job_id}")
    operator_status_policy = registry.policy_for("GET", "/api/studio/operator/status")
    audit_export_policy = registry.policy_for("GET", "/api/studio/audit/export")
    quarantine_export_policy = registry.policy_for(
        "GET",
        "/api/studio/audit/quarantine/export",
    )
    quarantine_archive_policy = registry.policy_for(
        "POST",
        "/api/studio/audit/quarantine/archive",
    )
    quarantine_archive_validate_policy = registry.policy_for(
        "POST",
        "/api/studio/audit/quarantine/archive/validate",
    )
    quarantine_archive_retention_policy = registry.policy_for(
        "GET",
        "/api/studio/audit/quarantine/archive/retention",
    )
    quarantine_archive_restore_policy = registry.policy_for(
        "POST",
        "/api/studio/audit/quarantine/archive/restore",
    )
    quarantine_archive_purge_policy = registry.policy_for(
        "POST",
        "/api/studio/audit/quarantine/archive/purge",
    )
    browser_user_create_policy = registry.policy_for(
        "POST",
        "/api/studio/identity/browser-users",
    )
    artifact_policy = registry.policy_for(
        "GET",
        "/api/studio/jobs/{job_id}/artifacts/{artifact_path:path}",
    )

    assert health_policy.visibility is contract["RouteVisibility"].PUBLIC
    assert capability_policy.visibility is contract["RouteVisibility"].PUBLIC
    assert detail_policy.visibility is contract["RouteVisibility"].PUBLIC
    assert jobs_list_policy.visibility is contract["RouteVisibility"].ADMIN
    assert jobs_list_policy.audit_action == "studio.jobs.list"
    assert job_detail_policy.visibility is contract["RouteVisibility"].ADMIN
    assert job_detail_policy.audit_action == "studio.jobs.detail"
    assert operator_status_policy.visibility is contract["RouteVisibility"].ADMIN
    assert audit_export_policy.visibility is contract["RouteVisibility"].ADMIN
    assert quarantine_export_policy.visibility is contract["RouteVisibility"].ADMIN
    assert quarantine_export_policy.audit_action == "studio.audit.quarantine.export"
    assert quarantine_archive_policy.visibility is contract["RouteVisibility"].ADMIN
    assert quarantine_archive_policy.audit_action == "studio.audit.quarantine.archive"
    assert quarantine_archive_validate_policy.visibility is contract["RouteVisibility"].ADMIN
    assert (
        quarantine_archive_validate_policy.audit_action
        == "studio.audit.quarantine.archive.validate"
    )
    assert quarantine_archive_retention_policy.visibility is contract["RouteVisibility"].ADMIN
    assert (
        quarantine_archive_retention_policy.audit_action
        == "studio.audit.quarantine.archive.retention"
    )
    assert quarantine_archive_restore_policy.visibility is contract["RouteVisibility"].ADMIN
    assert (
        quarantine_archive_restore_policy.audit_action
        == "studio.audit.quarantine.archive.restore"
    )
    assert quarantine_archive_purge_policy.visibility is contract["RouteVisibility"].ADMIN
    assert (
        quarantine_archive_purge_policy.audit_action
        == "studio.audit.quarantine.archive.purge"
    )
    assert browser_user_create_policy.visibility is contract["RouteVisibility"].ADMIN
    assert (
        browser_user_create_policy.audit_action
        == "studio.identity.browser_users.create"
    )
    assert artifact_policy.visibility is contract["RouteVisibility"].ADMIN
    assert artifact_policy.audit_action == "studio.jobs.artifact.read"


def test_default_route_policy_registry_reports_unclassified_platform_route() -> None:
    contract = _policy_contract()
    registry = contract["build_default_studio_route_policy_registry"]()

    missing = registry.missing_policies(
        (
            ("GET", "/api/health"),
            ("GET", "/api/studio/capabilities"),
            ("POST", "/api/studio/admin"),
        )
    )

    assert missing == ("POST /api/studio/admin",)


def test_route_policy_registry_exports_stable_policy_inventory() -> None:
    contract = _policy_contract()
    registry = contract["RoutePolicyRegistry"]()
    health_policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].PUBLIC,
        audit_action="studio.health.read",
    )
    admin_policy = contract["RoutePolicy"](
        visibility=contract["RouteVisibility"].ADMIN,
        audit_action="studio.admin.write",
    )

    registry.register("POST", "/api/studio/admin", admin_policy)
    registry.register("GET", "/api/health", health_policy)

    assert registry.policies() == (
        ("GET", "/api/health", health_policy),
        ("POST", "/api/studio/admin", admin_policy),
    )


def test_studio_app_exposes_route_policy_registry_for_platform_routes() -> None:
    from sc_neurocore.studio.app import create_app  # noqa: PLC0415
    from starlette.routing import Route  # noqa: PLC0415

    app = create_app()
    platform_routes: list[tuple[str, str]] = []
    for route in app.routes:
        if not isinstance(route, Route):
            continue
        if route.path != "/api/health" and not route.path.startswith("/api/studio/"):
            continue
        route_methods = route.methods or set()
        platform_routes.extend(
            (method, route.path) for method in sorted(route_methods) if method != "HEAD"
        )

    missing = app.state.studio_route_policies.missing_policies(tuple(platform_routes))

    assert missing == ()


def test_studio_app_classifies_every_api_and_websocket_route() -> None:
    from sc_neurocore.studio.app import create_app  # noqa: PLC0415
    from starlette.routing import Route, WebSocketRoute  # noqa: PLC0415

    app = create_app()
    route_signatures: list[tuple[str, str]] = []
    for route in app.routes:
        if isinstance(route, Route) and route.path.startswith("/api/"):
            route_methods = route.methods or set()
            route_signatures.extend(
                (method, route.path) for method in sorted(route_methods) if method != "HEAD"
            )
        elif isinstance(route, WebSocketRoute) and route.path.startswith("/ws/"):
            route_signatures.append(("WEBSOCKET", route.path))

    missing = app.state.studio_route_policies.missing_policies(tuple(route_signatures))

    assert missing == ()


def test_default_route_policy_registry_marks_stateful_routes_protected() -> None:
    contract = _policy_contract()
    registry = contract["build_default_studio_route_policy_registry"]()

    training_policy = registry.policy_for("POST", "/api/training/start")
    training_checkpoint_export_policy = registry.policy_for(
        "GET",
        "/api/training/checkpoint/{job_id}",
    )
    training_checkpoint_import_policy = registry.policy_for(
        "POST",
        "/api/training/checkpoint/import",
    )
    synth_policy = registry.policy_for("POST", "/api/synth/run")
    websocket_policy = registry.policy_for("WEBSOCKET", "/ws/progress")
    jobs_status_policy = registry.policy_for("GET", "/api/studio/jobs/status")
    jobs_list_policy = registry.policy_for("GET", "/api/studio/jobs")
    job_detail_policy = registry.policy_for("GET", "/api/studio/jobs/{job_id}")
    artifact_policy = registry.policy_for(
        "GET",
        "/api/studio/jobs/{job_id}/artifacts/{artifact_path:path}",
    )

    assert training_policy.visibility is contract["RouteVisibility"].AUTHENTICATED
    assert (
        training_checkpoint_export_policy.visibility
        is contract["RouteVisibility"].AUTHENTICATED
    )
    assert (
        training_checkpoint_import_policy.visibility
        is contract["RouteVisibility"].AUTHENTICATED
    )
    assert synth_policy.visibility is contract["RouteVisibility"].ADMIN
    assert websocket_policy.visibility is contract["RouteVisibility"].AUTHENTICATED
    assert jobs_status_policy.visibility is contract["RouteVisibility"].PUBLIC
    assert jobs_list_policy.visibility is contract["RouteVisibility"].ADMIN
    assert job_detail_policy.visibility is contract["RouteVisibility"].ADMIN
    assert artifact_policy.visibility is contract["RouteVisibility"].ADMIN
