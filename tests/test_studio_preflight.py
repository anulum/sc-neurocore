# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio preflight tests

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

import sc_neurocore.studio.platform.preflight as preflight
from sc_neurocore.studio.platform.bootstrap import bootstrap_studio_admin_identity
from sc_neurocore.studio.platform.identity import (
    add_studio_browser_user_record,
    update_studio_identity_record,
)
from sc_neurocore.studio.platform.policy import RouteVisibility
from sc_neurocore.studio.platform.preflight import (
    STUDIO_PREFLIGHT_SCHEMA_VERSION,
    StudioPreflightCheck,
    run_studio_preflight,
)


def _release_env(tmp_path: Path, identity_path: Path) -> dict[str, str]:
    return {
        "SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL": "false",
        "SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH": str(tmp_path / "audit" / "studio.jsonl"),
        "SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE": "production",
        "SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES": "true",
        "SC_NEUROCORE_STUDIO_IDENTITY_FILE": str(identity_path),
        "SC_NEUROCORE_STUDIO_JOB_ROOT": str(tmp_path / "jobs"),
    }


def _check_by_id(
    checks: tuple[StudioPreflightCheck, ...],
    check_id: str,
) -> StudioPreflightCheck:
    for check in checks:
        if check.check_id == check_id:
            return check
    raise AssertionError(f"Missing preflight check {check_id!r}")


def test_studio_preflight_passes_release_posture_without_secret_leaks(tmp_path: Path) -> None:
    identity_path = tmp_path / "private" / "studio-identities.json"
    bootstrap = bootstrap_studio_admin_identity(
        identity_path,
        token_factory=lambda _: "release-preflight-token",
    )
    (tmp_path / "audit").mkdir()
    (tmp_path / "jobs").mkdir()

    report = run_studio_preflight(_release_env(tmp_path, identity_path))
    payload = report.to_public_dict()
    encoded_payload = json.dumps(payload)

    assert report.passed is True
    assert payload["schema_version"] == STUDIO_PREFLIGHT_SCHEMA_VERSION
    assert payload["deployment_profile"] == "production"
    assert _check_by_id(report.checks, "identity_store").evidence["active_admin_principals"] == 1
    assert all(check.remediation == () for check in report.checks)
    assert bootstrap.bearer_token not in encoded_payload
    assert bootstrap.token_sha256 not in encoded_payload
    assert str(tmp_path) not in encoded_payload


def test_studio_preflight_fails_closed_for_development_defaults() -> None:
    report = run_studio_preflight({})
    failed_ids = {check.check_id for check in report.checks if check.status == "fail"}

    assert report.passed is False
    assert {
        "audit_log",
        "header_principal_fallback",
        "identity_store",
        "job_root",
        "route_policy_enforcement",
    }.issubset(failed_ids)
    for check in report.checks:
        if check.status == "fail":
            assert check.remediation
    assert _check_by_id(report.checks, "route_policy_inventory").status == "pass"


def test_studio_preflight_fails_on_invalid_runtime_settings() -> None:
    report = run_studio_preflight({"SC_NEUROCORE_STUDIO_CORS_ORIGINS": "*"})

    assert report.passed is False
    assert report.deployment_profile is None
    assert report.checks == (
        StudioPreflightCheck(
            check_id="runtime_settings",
            status="fail",
            message="Studio runtime settings reject wildcard CORS origins.",
            evidence={},
            remediation=(
                "Fix the invalid SC_NEUROCORE_STUDIO_* environment value.",
                "Regenerate a deployment package with sc-neurocore studio-deployment-profile.",
            ),
        ),
    )


def test_studio_preflight_detects_required_route_policy_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity_path = tmp_path / "studio-identities.json"
    bootstrap_studio_admin_identity(
        identity_path,
        token_factory=lambda _: "release-preflight-token",
    )
    (tmp_path / "audit").mkdir()
    (tmp_path / "jobs").mkdir()
    monkeypatch.setattr(
        preflight,
        "_REQUIRED_ROUTE_POLICIES",
        (
            ("GET", "/api/studio/missing", RouteVisibility.ADMIN, "studio.missing"),
            (
                "GET",
                "/api/studio/operator/status",
                RouteVisibility.PUBLIC,
                "studio.operator.status.changed",
            ),
        ),
    )

    report = run_studio_preflight(_release_env(tmp_path, identity_path))
    route_check = _check_by_id(report.checks, "route_policy_inventory")

    assert report.passed is False
    assert route_check.status == "fail"
    assert route_check.evidence["missing_count"] == 1
    assert route_check.evidence["mismatched_count"] == 1
    assert "route policy registry" in route_check.remediation[0]


def test_studio_preflight_reports_invalid_identity_without_path_leak(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    identity_path.write_text('{"schema_version":"wrong"}\n', encoding="utf-8")
    (tmp_path / "audit").mkdir()
    (tmp_path / "jobs").mkdir()

    report = run_studio_preflight(_release_env(tmp_path, identity_path))
    identity_check = _check_by_id(report.checks, "identity_store")
    encoded_payload = json.dumps(report.to_public_dict())

    assert report.passed is False
    assert identity_check.status == "fail"
    assert identity_check.evidence["configured"] is True
    assert identity_check.evidence["valid"] is False
    assert "identity file" in identity_check.remediation[0]
    assert str(identity_path) not in encoded_payload


def test_studio_preflight_counts_active_browser_admin(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    bootstrap_studio_admin_identity(
        identity_path,
        token_factory=lambda _: "release-preflight-token",
    )
    update_studio_identity_record(
        identity_path,
        principal_id="svc-studio-admin",
        roles=("studio.viewer",),
        active=True,
        expires_at_utc=None,
    )
    add_studio_browser_user_record(
        identity_path,
        username="viewer",
        principal_id="human-viewer",
        roles=("studio.viewer",),
        password="viewer-secret",
    )
    add_studio_browser_user_record(
        identity_path,
        username="operator",
        principal_id="human-operator",
        roles=("studio.admin",),
        password="operator-secret",
    )
    (tmp_path / "audit").mkdir()
    (tmp_path / "jobs").mkdir()

    report = run_studio_preflight(_release_env(tmp_path, identity_path))
    identity_check = _check_by_id(report.checks, "identity_store")

    assert report.passed is True
    assert identity_check.evidence["active_admin_principals"] == 1
    assert identity_check.evidence["browser_user_count"] == 2


def test_studio_preflight_fails_when_admin_principal_is_expired(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    bootstrap_studio_admin_identity(
        identity_path,
        token_factory=lambda _: "expired-release-token",
        expires_at_utc="2020-01-01T00:00:00+00:00",
    )
    (tmp_path / "audit").mkdir()
    (tmp_path / "jobs").mkdir()

    report = run_studio_preflight(_release_env(tmp_path, identity_path))
    identity_check = _check_by_id(report.checks, "identity_store")

    assert report.passed is False
    assert identity_check.status == "fail"
    assert identity_check.evidence["active_admin_principals"] == 0
    assert "studio.admin" in identity_check.remediation[0]


def test_studio_preflight_rejects_audit_log_directory(tmp_path: Path) -> None:
    identity_path = tmp_path / "studio-identities.json"
    bootstrap_studio_admin_identity(
        identity_path,
        token_factory=lambda _: "release-preflight-token",
    )
    audit_target = tmp_path / "audit-as-directory"
    audit_target.mkdir()
    env = _release_env(tmp_path, identity_path)
    env["SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH"] = str(audit_target)

    report = run_studio_preflight(env)
    audit_check = _check_by_id(report.checks, "audit_log")

    assert report.passed is False
    assert audit_check.status == "fail"
    assert audit_check.evidence["target_is_directory"] is True
    assert "audit-log parent" in audit_check.remediation[0]


def test_studio_preflight_public_remediation_is_path_and_secret_free() -> None:
    report = run_studio_preflight({})
    payload = report.to_public_dict()
    checks_payload = cast(list[dict[str, object]], payload["checks"])
    encoded_payload = json.dumps(payload).lower()
    identity_check = _check_by_id(report.checks, "identity_store")
    audit_check = _check_by_id(report.checks, "audit_log")
    job_check = _check_by_id(report.checks, "job_root")

    assert "remediation" in checks_payload[0]
    assert "sc_neurocore_studio_identity_file" in encoded_payload
    assert "password" not in encoded_payload
    assert "token_sha256" not in encoded_payload
    assert "bearer_token" not in encoded_payload
    assert "/home/" not in encoded_payload
    assert "/media/" not in encoded_payload
    assert identity_check.remediation[0].startswith("Create the first admin identity")
    assert audit_check.remediation[0].startswith("Set SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH")
    assert job_check.remediation[0].startswith("Set SC_NEUROCORE_STUDIO_JOB_ROOT")
    assert (
        preflight._environment_variable_for_check("identity_store")
        == "SC_NEUROCORE_STUDIO_IDENTITY_FILE"
    )
