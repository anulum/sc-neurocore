# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio policy audit rotation tests

"""Audit rotation tests for Studio policy."""

from __future__ import annotations

import json
from pathlib import Path

from tests.studio_policy_support import audit_event_hash, policy_contract


def test_jsonl_audit_sink_rotates_and_retains_hash_chain(tmp_path: Path) -> None:
    contract = policy_contract()
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
    assert current_row["event_hash"] == audit_event_hash(current_row)
    assert audit_sink.status().integrity_verified is True


def test_jsonl_audit_sink_starts_chain_after_blank_log(tmp_path: Path) -> None:
    contract = policy_contract()
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
    assert row["event_hash"] == audit_event_hash(row)


def test_jsonl_audit_sink_starts_chain_after_legacy_row(tmp_path: Path) -> None:
    contract = policy_contract()
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
    assert row["event_hash"] == audit_event_hash(row)
