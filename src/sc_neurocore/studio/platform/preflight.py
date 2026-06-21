# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio release preflight

"""Release-readiness preflight checks for SC-NeuroCore Studio."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TypeAlias

from sc_neurocore.studio.platform.identity import (
    StudioBrowserUserRecord,
    StudioIdentityRecord,
    StudioIdentityStore,
    load_studio_identity_store,
)
from sc_neurocore.studio.platform.policy import (
    RoutePolicy,
    RouteVisibility,
    build_default_studio_route_policy_registry,
)
from sc_neurocore.studio.platform.settings import (
    StudioRuntimeSettings,
    build_default_studio_runtime_settings,
)

STUDIO_PREFLIGHT_SCHEMA_VERSION = "studio.preflight.v1"
UTC = timezone.utc

StudioPreflightStatus: TypeAlias = Literal["pass", "fail", "warn"]
StudioPreflightEvidenceValue: TypeAlias = bool | float | int | str | None

_REQUIRED_ROUTE_POLICIES: tuple[tuple[str, str, RouteVisibility, str], ...] = (
    ("GET", "/api/studio/operator/status", RouteVisibility.ADMIN, "studio.operator.status.read"),
    ("GET", "/api/studio/audit/export", RouteVisibility.ADMIN, "studio.audit.export"),
    (
        "GET",
        "/api/studio/audit/quarantine/export",
        RouteVisibility.ADMIN,
        "studio.audit.quarantine.export",
    ),
    (
        "POST",
        "/api/studio/audit/quarantine/archive",
        RouteVisibility.ADMIN,
        "studio.audit.quarantine.archive",
    ),
    (
        "POST",
        "/api/studio/audit/quarantine/archive/validate",
        RouteVisibility.ADMIN,
        "studio.audit.quarantine.archive.validate",
    ),
    (
        "GET",
        "/api/studio/audit/quarantine/archive/retention",
        RouteVisibility.ADMIN,
        "studio.audit.quarantine.archive.retention",
    ),
    (
        "POST",
        "/api/studio/audit/quarantine/archive/restore",
        RouteVisibility.ADMIN,
        "studio.audit.quarantine.archive.restore",
    ),
    (
        "POST",
        "/api/studio/audit/quarantine/archive/purge",
        RouteVisibility.ADMIN,
        "studio.audit.quarantine.archive.purge",
    ),
    (
        "GET",
        "/api/studio/identity/service-accounts",
        RouteVisibility.ADMIN,
        "studio.identity.service_accounts.list",
    ),
    (
        "GET",
        "/api/studio/identity/service-accounts/{principal_id}",
        RouteVisibility.ADMIN,
        "studio.identity.service_accounts.detail",
    ),
    (
        "PATCH",
        "/api/studio/identity/service-accounts/{principal_id}",
        RouteVisibility.ADMIN,
        "studio.identity.service_accounts.update",
    ),
    (
        "GET",
        "/api/studio/identity/browser-users",
        RouteVisibility.ADMIN,
        "studio.identity.browser_users.list",
    ),
    (
        "POST",
        "/api/studio/identity/browser-users",
        RouteVisibility.ADMIN,
        "studio.identity.browser_users.create",
    ),
    (
        "GET",
        "/api/studio/identity/browser-users/{username}",
        RouteVisibility.ADMIN,
        "studio.identity.browser_users.detail",
    ),
    (
        "PATCH",
        "/api/studio/identity/browser-users/{username}",
        RouteVisibility.ADMIN,
        "studio.identity.browser_users.update",
    ),
    (
        "POST",
        "/api/studio/identity/browser-users/{username}/password",
        RouteVisibility.ADMIN,
        "studio.identity.browser_users.password.rotate",
    ),
    ("GET", "/api/studio/jobs", RouteVisibility.ADMIN, "studio.jobs.list"),
    ("GET", "/api/studio/jobs/{job_id}", RouteVisibility.ADMIN, "studio.jobs.detail"),
    (
        "GET",
        "/api/studio/jobs/{job_id}/artifacts/{artifact_path:path}",
        RouteVisibility.ADMIN,
        "studio.jobs.artifact.read",
    ),
    ("POST", "/api/studio/evidence/bundle", RouteVisibility.ADMIN, "studio.evidence.bundle.create"),
    (
        "POST",
        "/api/studio/training/weight-restore",
        RouteVisibility.ADMIN,
        "studio.training.weight_restore.materialize",
    ),
    (
        "POST",
        "/api/studio/training/weight-restore/attach",
        RouteVisibility.ADMIN,
        "studio.training.weight_restore.attach",
    ),
    ("POST", "/api/synth/run", RouteVisibility.ADMIN, "studio.synth.run"),
    ("POST", "/api/synth/pnr", RouteVisibility.ADMIN, "studio.synth.pnr"),
)


@dataclass(frozen=True, slots=True)
class StudioPreflightCheck:
    """One secret-free Studio release-preflight check result.

    Parameters
    ----------
    check_id:
        Stable machine-readable identifier for the checked release invariant.
    status:
        ``"pass"`` when the invariant holds, ``"fail"`` when it is violated and
        blocks release, or ``"warn"`` for a non-blocking advisory the operator
        should resolve before production use.
    message:
        Operator-facing summary that does not include local paths or secrets.
    evidence:
        Small scalar evidence fields suitable for JSON reports.
    remediation:
        Operator actions that can resolve a failed or warned check. Entries
        must not expose local filesystem paths or secret material.
    """

    check_id: str
    status: StudioPreflightStatus
    message: str
    evidence: Mapping[str, StudioPreflightEvidenceValue] = field(default_factory=dict)
    remediation: tuple[str, ...] = ()

    def to_public_dict(self) -> dict[str, object]:
        """Return a JSON-serializable, path-free check payload."""

        return {
            "check_id": self.check_id,
            "evidence": dict(sorted(self.evidence.items())),
            "message": self.message,
            "remediation": list(self.remediation),
            "status": self.status,
        }


@dataclass(frozen=True, slots=True)
class StudioPreflightReport:
    """Machine-readable Studio release-preflight report.

    Parameters
    ----------
    checks:
        Ordered release-readiness checks.
    deployment_profile:
        Parsed Studio deployment profile when runtime settings were valid.
    schema_version:
        Stable report schema identifier.
    """

    checks: tuple[StudioPreflightCheck, ...]
    deployment_profile: str | None
    schema_version: str = STUDIO_PREFLIGHT_SCHEMA_VERSION

    @property
    def passed(self) -> bool:
        """Return whether no preflight check failed (warnings do not block)."""

        return all(check.status != "fail" for check in self.checks)

    @property
    def warned(self) -> bool:
        """Return whether any check raised a non-blocking advisory warning."""

        return any(check.status == "warn" for check in self.checks)

    def to_public_dict(self) -> dict[str, object]:
        """Return a JSON-serializable, secret-free preflight payload."""

        return {
            "checks": [check.to_public_dict() for check in self.checks],
            "deployment_profile": self.deployment_profile,
            "passed": self.passed,
            "schema_version": self.schema_version,
            "warned": self.warned,
        }


def run_studio_preflight(
    env: Mapping[str, str] | None = None,
    *,
    clock: datetime | None = None,
) -> StudioPreflightReport:
    """Run Studio release-readiness checks from environment-style settings.

    Parameters
    ----------
    env:
        Optional environment mapping. When omitted, ``os.environ`` is used by
        the runtime settings builder.
    clock:
        Optional UTC timestamp used for expiry-sensitive identity checks.

    Returns
    -------
    StudioPreflightReport
        Ordered path-free report. The report fails closed if settings cannot
        be parsed or any release invariant is missing.
    """

    checks: list[StudioPreflightCheck] = []
    try:
        settings = build_default_studio_runtime_settings(env)
    except ValueError as exc:
        return StudioPreflightReport(
            checks=(
                StudioPreflightCheck(
                    check_id="runtime_settings",
                    status="fail",
                    message=str(exc),
                    evidence={},
                    remediation=(
                        "Fix the invalid SC_NEUROCORE_STUDIO_* environment value.",
                        "Regenerate a deployment package with sc-neurocore studio-deployment-profile.",
                    ),
                ),
            ),
            deployment_profile=None,
        )

    checks.append(
        StudioPreflightCheck(
            check_id="runtime_settings",
            status="pass",
            message="Studio runtime settings parsed successfully.",
            evidence={"deployment_profile": settings.deployment_profile},
        )
    )
    checks.extend(_profile_checks(settings))
    checks.append(_browser_login_lockout_check(settings))
    checks.append(_route_policy_inventory_check())
    checks.append(_identity_store_check(settings, now=clock or datetime.now(UTC)))
    checks.append(
        _path_readiness_check(
            check_id="audit_log",
            configured_path=settings.audit_log_path,
            target_kind="file",
        )
    )
    checks.append(
        _path_readiness_check(
            check_id="job_root",
            configured_path=settings.job_root_path,
            target_kind="directory",
        )
    )
    checks.append(_resource_limits_check(settings))
    return StudioPreflightReport(
        checks=tuple(checks),
        deployment_profile=settings.deployment_profile,
    )


def _profile_checks(settings: StudioRuntimeSettings) -> tuple[StudioPreflightCheck, ...]:
    return (
        StudioPreflightCheck(
            check_id="route_policy_enforcement",
            status="pass" if settings.enforce_route_policies else "fail",
            message=(
                "Route-policy enforcement is enabled."
                if settings.enforce_route_policies
                else "Release preflight requires route-policy enforcement."
            ),
            evidence={"enforced": settings.enforce_route_policies},
            remediation=()
            if settings.enforce_route_policies
            else ("Set SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES=true.",),
        ),
        StudioPreflightCheck(
            check_id="header_principal_fallback",
            status="fail" if settings.allow_header_principal else "pass",
            message=(
                "Development header-principal fallback is disabled."
                if not settings.allow_header_principal
                else "Release preflight requires development header-principal fallback disabled."
            ),
            evidence={"allow_header_principal": settings.allow_header_principal},
            remediation=()
            if not settings.allow_header_principal
            else ("Set SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL=false.",),
        ),
    )


def _browser_login_lockout_check(settings: StudioRuntimeSettings) -> StudioPreflightCheck:
    passed = (
        settings.browser_login_max_failures > 0
        and settings.browser_login_failure_window_seconds > 0
        and settings.browser_login_cooldown_seconds > 0
    )
    return StudioPreflightCheck(
        check_id="browser_login_lockout",
        status="pass" if passed else "fail",
        message=(
            "Browser login lockout limits are configured."
            if passed
            else "Browser login lockout limits must be positive."
        ),
        evidence={
            "cooldown_seconds": settings.browser_login_cooldown_seconds,
            "failure_window_seconds": settings.browser_login_failure_window_seconds,
            "max_failures": settings.browser_login_max_failures,
        },
        remediation=()
        if passed
        else (
            "Set positive SC_NEUROCORE_STUDIO_BROWSER_LOGIN_* limits before launch.",
            "Rerun studio-preflight from the deployment environment.",
        ),
    )


def _eda_limits_enforceable() -> bool:
    """Return whether this host can enforce POSIX child-process resource limits.

    Mirrors ``sc_neurocore.studio.synthesis._eda_process_limits_supported``: the
    ``resource`` rlimit primitives Studio uses for Yosys and nextpnr child
    processes are only available on POSIX hosts.
    """

    return os.name == "posix"


def _resource_limits_check(settings: StudioRuntimeSettings) -> StudioPreflightCheck:
    cpu_seconds = settings.eda_process_cpu_seconds
    memory_bytes = settings.eda_process_memory_bytes
    enforceable = _eda_limits_enforceable()
    ceilings_configured = cpu_seconds is not None and memory_bytes is not None
    evidence: dict[str, StudioPreflightEvidenceValue] = {
        "eda_process_cpu_seconds": cpu_seconds,
        "eda_process_limits_enforceable": enforceable,
        "eda_process_memory_bytes": memory_bytes,
        "job_default_timeout_seconds": settings.job_default_timeout_seconds,
        "job_max_artifact_bytes": settings.job_max_artifact_bytes,
    }
    if not ceilings_configured:
        return StudioPreflightCheck(
            check_id="resource_limits",
            status="warn",
            message="EDA process CPU and memory ceilings are not configured; child processes run unbounded.",
            evidence=evidence,
            remediation=(
                "Set SC_NEUROCORE_STUDIO_EDA_PROCESS_CPU_SECONDS and "
                "SC_NEUROCORE_STUDIO_EDA_PROCESS_MEMORY_BYTES before launch.",
                "Use the server deployment profile for default bounded EDA ceilings.",
            ),
        )
    if not enforceable:
        return StudioPreflightCheck(
            check_id="resource_limits",
            status="warn",
            message="EDA process ceilings are configured but cannot be enforced on this host.",
            evidence=evidence,
            remediation=(
                "Deploy Studio on a POSIX host so configured EDA ceilings apply.",
                "Otherwise bound Yosys and nextpnr through the container or orchestrator runtime.",
            ),
        )
    return StudioPreflightCheck(
        check_id="resource_limits",
        status="pass",
        message="EDA process ceilings and the job artifact size limit are configured and enforceable.",
        evidence=evidence,
    )


def _route_policy_inventory_check() -> StudioPreflightCheck:
    registry = build_default_studio_route_policy_registry()
    missing: list[str] = []
    mismatched: list[str] = []
    for method, path_template, visibility, audit_action in _REQUIRED_ROUTE_POLICIES:
        try:
            policy = registry.policy_for(method, path_template)
        except KeyError:
            missing.append(f"{method} {path_template}")
            continue
        if _route_policy_mismatch(policy, visibility=visibility, audit_action=audit_action):
            mismatched.append(f"{method} {path_template}")
    failure_count = len(missing) + len(mismatched)
    return StudioPreflightCheck(
        check_id="route_policy_inventory",
        status="pass" if failure_count == 0 else "fail",
        message=(
            "Required Studio release routes have fail-closed policies."
            if failure_count == 0
            else "One or more required Studio release routes are missing fail-closed policies."
        ),
        evidence={
            "required_route_count": len(_REQUIRED_ROUTE_POLICIES),
            "missing_count": len(missing),
            "mismatched_count": len(mismatched),
        },
        remediation=()
        if failure_count == 0
        else (
            "Update the Studio route policy registry for every protected route.",
            "Rerun the Studio route-policy tests before deployment.",
        ),
    )


def _route_policy_mismatch(
    policy: RoutePolicy,
    *,
    visibility: RouteVisibility,
    audit_action: str,
) -> bool:
    return policy.visibility is not visibility or policy.audit_action != audit_action


def _identity_store_check(
    settings: StudioRuntimeSettings,
    *,
    now: datetime,
) -> StudioPreflightCheck:
    if settings.identity_file_path is None:
        return StudioPreflightCheck(
            check_id="identity_store",
            status="fail",
            message="Release preflight requires a configured Studio identity file.",
            evidence={"configured": False},
            remediation=(
                "Create the first admin identity with sc-neurocore studio-bootstrap-admin.",
                "Set SC_NEUROCORE_STUDIO_IDENTITY_FILE to the identity file location.",
            ),
        )
    try:
        store = load_studio_identity_store(Path(settings.identity_file_path))
    except ValueError as exc:
        return StudioPreflightCheck(
            check_id="identity_store",
            status="fail",
            message=str(exc),
            evidence={"configured": True, "valid": False},
            remediation=(
                "Replace the identity file with a valid sc-neurocore.studio.identity.v1 document.",
                "Store raw operator secrets outside repository files.",
            ),
        )
    active_admin_count = _active_admin_principal_count(store, now=now.astimezone(UTC))
    passed = active_admin_count > 0
    return StudioPreflightCheck(
        check_id="identity_store",
        status="pass" if passed else "fail",
        message=(
            "Identity store has at least one active unexpired admin principal."
            if passed
            else "Release preflight requires at least one active unexpired admin principal."
        ),
        evidence={
            "active_admin_principals": active_admin_count,
            "browser_user_count": len(store.browser_users),
            "configured": True,
            "service_account_count": len(store.service_accounts),
            "valid": True,
        },
        remediation=()
        if passed
        else (
            "Enable or create at least one unexpired principal with the studio.admin role.",
            "Rotate expired or disabled identities before deployment.",
        ),
    )


def _active_admin_principal_count(
    store: StudioIdentityStore,
    *,
    now: datetime,
) -> int:
    service_admins = sum(
        1 for record in store.service_accounts if _active_admin_identity(record, now=now)
    )
    browser_admins = sum(
        1 for record in store.browser_users if _active_admin_browser_user(record, now=now)
    )
    return service_admins + browser_admins


def _active_admin_identity(record: StudioIdentityRecord, *, now: datetime) -> bool:
    if not record.active or "studio.admin" not in record.roles:
        return False
    return record.expires_at_utc is None or now < record.expires_at_utc


def _active_admin_browser_user(record: StudioBrowserUserRecord, *, now: datetime) -> bool:
    if not record.active or "studio.admin" not in record.roles:
        return False
    return record.expires_at_utc is None or now < record.expires_at_utc


def _path_readiness_check(
    *,
    check_id: str,
    configured_path: str | None,
    target_kind: Literal["directory", "file"],
) -> StudioPreflightCheck:
    if configured_path is None:
        return StudioPreflightCheck(
            check_id=check_id,
            status="fail",
            message=f"Release preflight requires configured Studio {check_id.replace('_', ' ')}.",
            evidence={"configured": False},
            remediation=(
                f"Set {_environment_variable_for_check(check_id)} to a durable deployment path.",
                "Include that location in the Studio backup and restore plan.",
            ),
        )
    path = Path(configured_path)
    if target_kind == "file":
        return _file_path_readiness_check(check_id=check_id, path=path)
    return _directory_path_readiness_check(check_id=check_id, path=path)


def _file_path_readiness_check(*, check_id: str, path: Path) -> StudioPreflightCheck:
    path_exists = path.exists()
    parent_exists = path.parent.exists()
    parent_is_directory = path.parent.is_dir()
    path_is_directory = path_exists and path.is_dir()
    parent_writable = parent_is_directory and os.access(path.parent, os.W_OK | os.X_OK)
    passed = not path_is_directory and parent_exists and parent_is_directory and parent_writable
    return StudioPreflightCheck(
        check_id=check_id,
        status="pass" if passed else "fail",
        message=(
            "Studio audit log location is ready for append-only JSONL output."
            if passed
            else "Studio audit log parent must exist, be writable, and the target must not be a directory."
        ),
        evidence={
            "configured": True,
            "parent_exists": parent_exists,
            "parent_is_directory": parent_is_directory,
            "parent_writable": parent_writable,
            "target_exists": path_exists,
            "target_is_directory": path_is_directory,
        },
        remediation=()
        if passed
        else (
            "Create a writable audit-log parent directory before launch.",
            "Ensure the configured audit-log target is a file path, not a directory.",
        ),
    )


def _directory_path_readiness_check(*, check_id: str, path: Path) -> StudioPreflightCheck:
    target_exists = path.exists()
    parent_exists = path.parent.exists()
    parent_is_directory = path.parent.is_dir()
    target_is_directory = target_exists and path.is_dir()
    parent_writable = parent_is_directory and os.access(path.parent, os.W_OK | os.X_OK)
    passed = (target_exists and target_is_directory) or (
        not target_exists and parent_exists and parent_is_directory and parent_writable
    )
    return StudioPreflightCheck(
        check_id=check_id,
        status="pass" if passed else "fail",
        message=(
            "Studio job root is ready for path-confined worker directories."
            if passed
            else "Studio job root must be an existing directory or have a writable parent."
        ),
        evidence={
            "configured": True,
            "parent_exists": parent_exists,
            "parent_is_directory": parent_is_directory,
            "parent_writable": parent_writable,
            "target_exists": target_exists,
            "target_is_directory": target_is_directory,
        },
        remediation=()
        if passed
        else (
            "Create the Studio job-root directory or a writable parent directory before launch.",
            "Place the job root on storage included in the Studio backup plan.",
        ),
    )


def _environment_variable_for_check(check_id: str) -> str:
    if check_id == "audit_log":
        return "SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH"
    if check_id == "job_root":
        return "SC_NEUROCORE_STUDIO_JOB_ROOT"
    return "SC_NEUROCORE_STUDIO_IDENTITY_FILE"
