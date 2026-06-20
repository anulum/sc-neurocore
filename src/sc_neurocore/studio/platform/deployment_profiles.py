# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio deployment profiles

"""Deployment-profile manifests for SC-NeuroCore Studio operators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

from sc_neurocore.studio.platform.settings import StudioDeploymentProfile

STUDIO_DEPLOYMENT_PROFILE_SCHEMA_VERSION = "studio.deployment-profile.v1"

StudioDeploymentPackageName: TypeAlias = Literal["local", "lab", "server"]


@dataclass(frozen=True, slots=True)
class StudioDeploymentProfilePackage:
    """Machine-readable Studio deployment profile package.

    Parameters
    ----------
    name:
        Operator-facing deployment package name.
    runtime_profile:
        Studio runtime profile value exported through
        ``SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE``.
    summary:
        Short profile summary for operator dashboards and release notes.
    environment:
        Environment variables required by this package. Values are placeholders
        or concrete safe defaults and must not contain secrets.
    required_operator_inputs:
        Secret or path values that the operator must provide outside the
        repository before launching Studio.
    security_controls:
        Controls that are active or required for the profile.
    backup_items:
        Durable state that must be included in local backup/restore plans.
    preflight_command:
        Command that validates the profile from the target launch environment.
    launch_command:
        Command used to launch the Studio process after environment setup.
    schema_version:
        Stable JSON schema identifier.
    """

    name: StudioDeploymentPackageName
    runtime_profile: StudioDeploymentProfile
    summary: str
    environment: Mapping[str, str]
    required_operator_inputs: tuple[str, ...]
    security_controls: tuple[str, ...]
    backup_items: tuple[str, ...]
    preflight_command: str
    launch_command: str
    schema_version: str = STUDIO_DEPLOYMENT_PROFILE_SCHEMA_VERSION

    def to_public_dict(self) -> dict[str, object]:
        """Return a JSON-serializable deployment package manifest."""

        return {
            "backup_items": list(self.backup_items),
            "environment": dict(sorted(self.environment.items())),
            "launch_command": self.launch_command,
            "name": self.name,
            "preflight_command": self.preflight_command,
            "required_operator_inputs": list(self.required_operator_inputs),
            "runtime_profile": self.runtime_profile,
            "schema_version": self.schema_version,
            "security_controls": list(self.security_controls),
            "summary": self.summary,
        }

    def to_env_lines(self) -> tuple[str, ...]:
        """Return sorted shell ``export`` lines for non-secret profile values."""

        return tuple(
            f"export {name}={_shell_quote(value)}"
            for name, value in sorted(self.environment.items())
        )


def build_studio_deployment_profile_package(
    name: StudioDeploymentPackageName,
) -> StudioDeploymentProfilePackage:
    """Build one Studio deployment-profile package.

    Parameters
    ----------
    name:
        Deployment package name. ``local`` targets single-operator loopback
        use, ``lab`` targets private LAN or VPN-hosted research workstations,
        and ``server`` targets reverse-proxied service deployment.

    Returns
    -------
    StudioDeploymentProfilePackage
        Path-placeholder manifest with environment, preflight, launch, and
        backup guidance.

    Raises
    ------
    ValueError
        If ``name`` is not a supported Studio deployment package.
    """

    if name == "local":
        return _local_profile_package()
    if name == "lab":
        return _lab_profile_package()
    if name == "server":
        return _server_profile_package()
    raise ValueError("Studio deployment package must be local, lab, or server.")


def list_studio_deployment_profile_packages() -> tuple[StudioDeploymentProfilePackage, ...]:
    """Return all supported Studio deployment-profile packages."""

    return (
        build_studio_deployment_profile_package("local"),
        build_studio_deployment_profile_package("lab"),
        build_studio_deployment_profile_package("server"),
    )


def _local_profile_package() -> StudioDeploymentProfilePackage:
    return StudioDeploymentProfilePackage(
        name="local",
        runtime_profile="development",
        summary="Loopback-only single-operator Studio for workstation exploration.",
        environment={
            "SC_NEUROCORE_STUDIO_ALLOWED_HOSTS": "127.0.0.1,localhost",
            "SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL": "true",
            "SC_NEUROCORE_STUDIO_CORS_ORIGINS": (
                "http://127.0.0.1:8001,http://localhost:8001,"
                "http://127.0.0.1:5173,http://localhost:5173"
            ),
            "SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE": "development",
            "SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES": "false",
            "SC_NEUROCORE_STUDIO_JOB_ROOT": "<local-job-root>",
            "SC_NEUROCORE_STUDIO_WEBSOCKET_ALLOWED_ORIGINS": (
                "http://127.0.0.1:8001,http://localhost:8001,"
                "http://127.0.0.1:5173,http://localhost:5173"
            ),
        },
        required_operator_inputs=("<local-job-root>",),
        security_controls=(
            "loopback hosts only",
            "no wildcard CORS or WebSocket origins",
            "development header principal allowed for local-only use",
            "persistent job root recommended",
        ),
        backup_items=("<local-job-root>", "<studio-project-root>"),
        preflight_command="sc-neurocore studio-preflight",
        launch_command="sc-neurocore studio --port 8001",
    )


def _lab_profile_package() -> StudioDeploymentProfilePackage:
    return StudioDeploymentProfilePackage(
        name="lab",
        runtime_profile="production",
        summary="Private lab workstation or VPN service with durable identity and audit state.",
        environment={
            "SC_NEUROCORE_STUDIO_ALLOWED_HOSTS": "<lab-hostname>,127.0.0.1,localhost",
            "SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL": "false",
            "SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH": "<audit-log-path>",
            "SC_NEUROCORE_STUDIO_AUDIT_RETAINED_FILES": "14",
            "SC_NEUROCORE_STUDIO_BROWSER_LOGIN_COOLDOWN_SECONDS": "900",
            "SC_NEUROCORE_STUDIO_BROWSER_LOGIN_FAILURE_WINDOW_SECONDS": "300",
            "SC_NEUROCORE_STUDIO_BROWSER_LOGIN_MAX_FAILURES": "5",
            "SC_NEUROCORE_STUDIO_CORS_ORIGINS": "https://<lab-hostname>",
            "SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE": "production",
            "SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES": "true",
            "SC_NEUROCORE_STUDIO_IDENTITY_FILE": "<identity-file>",
            "SC_NEUROCORE_STUDIO_JOB_ROOT": "<job-root>",
            "SC_NEUROCORE_STUDIO_WEBSOCKET_ALLOWED_ORIGINS": "https://<lab-hostname>",
        },
        required_operator_inputs=(
            "<lab-hostname>",
            "<identity-file>",
            "<audit-log-path>",
            "<job-root>",
            "browser-user or service-account secret material in an external secret store",
        ),
        security_controls=(
            "production runtime profile",
            "route-policy enforcement",
            "header principal fallback disabled",
            "durable identity file",
            "append-only audit log",
            "persistent job root",
            "explicit host and origin allow-lists",
        ),
        backup_items=(
            "<identity-file>",
            "<audit-log-path>",
            "<job-root>",
            "<studio-project-root>",
        ),
        preflight_command="sc-neurocore studio-preflight --output studio-preflight.json",
        launch_command="sc-neurocore studio --port 8001",
    )


def _server_profile_package() -> StudioDeploymentProfilePackage:
    return StudioDeploymentProfilePackage(
        name="server",
        runtime_profile="production",
        summary="Reverse-proxied Studio service with durable state and stricter limits.",
        environment={
            "SC_NEUROCORE_STUDIO_ALLOWED_HOSTS": "<public-studio-hostname>,127.0.0.1",
            "SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL": "false",
            "SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH": "<audit-log-path>",
            "SC_NEUROCORE_STUDIO_AUDIT_RETAINED_FILES": "30",
            "SC_NEUROCORE_STUDIO_AUDIT_ROTATION_BYTES": "104857600",
            "SC_NEUROCORE_STUDIO_BROWSER_LOGIN_COOLDOWN_SECONDS": "1800",
            "SC_NEUROCORE_STUDIO_BROWSER_LOGIN_FAILURE_WINDOW_SECONDS": "300",
            "SC_NEUROCORE_STUDIO_BROWSER_LOGIN_MAX_FAILURES": "5",
            "SC_NEUROCORE_STUDIO_CORS_ORIGINS": "https://<public-studio-hostname>",
            "SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE": "production",
            "SC_NEUROCORE_STUDIO_EDA_PROCESS_CPU_SECONDS": "120",
            "SC_NEUROCORE_STUDIO_EDA_PROCESS_MEMORY_BYTES": "2147483648",
            "SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES": "true",
            "SC_NEUROCORE_STUDIO_IDENTITY_FILE": "<identity-file>",
            "SC_NEUROCORE_STUDIO_JOB_MAX_ARTIFACT_BYTES": "16777216",
            "SC_NEUROCORE_STUDIO_JOB_ROOT": "<job-root>",
            "SC_NEUROCORE_STUDIO_JOB_TIMEOUT_SECONDS": "300",
            "SC_NEUROCORE_STUDIO_WEBSOCKET_ALLOWED_ORIGINS": (
                "https://<public-studio-hostname>"
            ),
        },
        required_operator_inputs=(
            "<public-studio-hostname>",
            "<identity-file>",
            "<audit-log-path>",
            "<job-root>",
            "TLS termination at the reverse proxy",
            "external backup target for durable Studio state",
            "service-account and browser-user secret material in an external secret store",
        ),
        security_controls=(
            "production runtime profile",
            "route-policy enforcement",
            "header principal fallback disabled",
            "durable identity file",
            "append-only rotated audit log",
            "persistent job root",
            "explicit host and origin allow-lists",
            "process CPU and memory ceilings for EDA jobs",
            "reverse proxy TLS required",
        ),
        backup_items=(
            "<identity-file>",
            "<audit-log-path>",
            "<job-root>",
            "<studio-project-root>",
        ),
        preflight_command="sc-neurocore studio-preflight --output studio-preflight.json",
        launch_command="sc-neurocore studio --port 8001",
    )


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


__all__ = [
    "STUDIO_DEPLOYMENT_PROFILE_SCHEMA_VERSION",
    "StudioDeploymentPackageName",
    "StudioDeploymentProfilePackage",
    "build_studio_deployment_profile_package",
    "list_studio_deployment_profile_packages",
]
