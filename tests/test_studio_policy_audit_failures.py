# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio policy audit failure handling tests

"""Audit failure handling tests for Studio policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.studio_policy_support import policy_contract


def test_jsonl_audit_sink_reports_failed_append_policy(tmp_path: Path) -> None:
    contract = policy_contract()
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
    contract = policy_contract()
    audit_sink = contract["JsonlAuditSink"](tmp_path)

    status = audit_sink.status()

    assert status.configured is True
    assert status.healthy is False
    assert status.last_error == "AuditPathIsDirectory"


def test_jsonl_audit_sink_status_rejects_file_parent(tmp_path: Path) -> None:
    contract = policy_contract()
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
    contract = policy_contract()
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
