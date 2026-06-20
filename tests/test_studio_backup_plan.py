# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio backup plan tests

"""Tests for Studio backup and restore planning manifests."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import pytest

from sc_neurocore.cli import main
from sc_neurocore.studio.platform.backup_plan import (
    STUDIO_BACKUP_PLAN_SCHEMA_VERSION,
    build_studio_backup_plan,
)
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings


def _production_settings(tmp_path: Path) -> StudioRuntimeSettings:
    identity_path = tmp_path / "identity" / "studio-identities.json"
    audit_path = tmp_path / "audit" / "studio.jsonl"
    job_root = tmp_path / "jobs"
    identity_path.parent.mkdir()
    audit_path.parent.mkdir()
    job_root.mkdir()
    identity_path.write_text('{"schema_version":"test"}\n', encoding="utf-8")
    audit_path.write_text("", encoding="utf-8")
    return StudioRuntimeSettings(
        deployment_profile="production",
        enforce_route_policies=True,
        allow_header_principal=False,
        identity_file_path=str(identity_path),
        audit_log_path=str(audit_path),
        job_root_path=str(job_root),
    )


def test_studio_backup_plan_reports_required_durable_state_without_paths(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "projects"
    project_root.mkdir()
    plan = build_studio_backup_plan(
        _production_settings(tmp_path),
        project_root=project_root,
    )
    payload = plan.to_public_dict()
    items = cast(list[dict[str, object]], payload["items"])
    encoded = json.dumps(payload)
    item_ids = {item["item_id"] for item in items}

    assert payload["schema_version"] == STUDIO_BACKUP_PLAN_SCHEMA_VERSION
    assert payload["deployment_profile"] == "production"
    assert payload["missing_required_count"] == 0
    assert payload["missing_existing_count"] == 0
    assert payload["ready_for_restore_drill"] is True
    assert item_ids == {"identity_file", "audit_log", "job_root", "project_workspace"}
    assert all("local_path" not in item for item in items)
    assert str(tmp_path) not in encoded
    assert "/home/" not in encoded
    assert "/media/" not in encoded
    assert "password" not in encoded.lower()
    assert "token_sha256" not in encoded
    assert "bearer_token" not in encoded


def test_studio_backup_plan_can_emit_local_paths_for_internal_handoff(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "projects"
    project_root.mkdir()
    plan = build_studio_backup_plan(
        _production_settings(tmp_path),
        include_local_paths=True,
        project_root=project_root,
    )
    payload = plan.to_public_dict()
    items_payload = cast(list[dict[str, object]], payload["items"])
    items = {str(item["item_id"]): item for item in items_payload}

    assert payload["include_local_paths"] is True
    assert str(items["identity_file"]["local_path"]).endswith("studio-identities.json")
    assert str(items["audit_log"]["local_path"]).endswith("studio.jsonl")
    assert str(items["job_root"]["local_path"]).endswith("jobs")
    assert str(items["project_workspace"]["local_path"]).endswith("projects")


def test_studio_backup_plan_marks_missing_production_targets() -> None:
    settings = StudioRuntimeSettings(
        deployment_profile="production",
        enforce_route_policies=True,
        allow_header_principal=False,
        identity_file_path="/tmp/missing-identity.json",
        audit_log_path="/tmp/missing-audit.jsonl",
        job_root_path="/tmp/missing-jobs",
    )
    plan = build_studio_backup_plan(settings, project_root=Path("/tmp/missing-projects"))
    payload = plan.to_public_dict()

    assert payload["missing_required_count"] == 0
    assert payload["missing_existing_count"] == 4
    assert payload["ready_for_restore_drill"] is False


def test_studio_backup_plan_cli_prints_path_free_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in (
        "SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE",
        "SC_NEUROCORE_STUDIO_IDENTITY_FILE",
        "SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH",
        "SC_NEUROCORE_STUDIO_JOB_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(sys, "argv", ["sc-neurocore", "studio-backup-plan"])

    exit_code = main()
    payload = json.loads(capsys.readouterr().out)
    encoded = json.dumps(payload)

    assert exit_code == 0
    assert payload["schema_version"] == STUDIO_BACKUP_PLAN_SCHEMA_VERSION
    assert payload["deployment_profile"] == "development"
    items = cast(list[dict[str, object]], payload["items"])
    assert all("local_path" not in item for item in items)
    assert "/home/" not in encoded
    assert "/media/" not in encoded


def test_studio_backup_plan_cli_writes_explicit_local_path_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "studio-backup-plan.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sc-neurocore",
            "studio-backup-plan",
            "--include-local-paths",
            "--output",
            str(output_path),
        ],
    )

    exit_code = main()
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert capsys.readouterr().out == ""
    assert payload["include_local_paths"] is True
    assert any("local_path" in item for item in payload["items"])
