# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio runtime settings

"""Runtime settings for SC-NeuroCore Studio."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

StudioDeploymentProfile = Literal["development", "production"]

DEFAULT_STUDIO_CORS_ORIGINS: tuple[str, ...] = (
    "http://127.0.0.1:8001",
    "http://localhost:8001",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
)

DEFAULT_STUDIO_WEBSOCKET_ALLOWED_ORIGINS = DEFAULT_STUDIO_CORS_ORIGINS

DEFAULT_STUDIO_ALLOWED_HOSTS: tuple[str, ...] = (
    "127.0.0.1",
    "localhost",
)

DEFAULT_STUDIO_MAX_REQUEST_BODY_BYTES = 1_048_576
DEFAULT_STUDIO_AUDIT_RETAINED_FILES = 5
DEFAULT_STUDIO_JOB_TIMEOUT_SECONDS = 300.0

DEFAULT_STUDIO_HTTP_SECURITY_HEADERS: Mapping[str, str] = MappingProxyType(
    {
        "x-content-type-options": "nosniff",
        "referrer-policy": "no-referrer",
        "x-frame-options": "DENY",
    }
)


def _default_studio_http_security_headers() -> Mapping[str, str]:
    """Return immutable default HTTP security headers for Studio responses."""

    return DEFAULT_STUDIO_HTTP_SECURITY_HEADERS


@dataclass(frozen=True, slots=True)
class StudioRuntimeSettings:
    """Runtime settings consumed by the Studio FastAPI application."""

    deployment_profile: StudioDeploymentProfile = "development"
    cors_allowed_origins: tuple[str, ...] = DEFAULT_STUDIO_CORS_ORIGINS
    websocket_allowed_origins: tuple[str, ...] = DEFAULT_STUDIO_WEBSOCKET_ALLOWED_ORIGINS
    allowed_hosts: tuple[str, ...] = DEFAULT_STUDIO_ALLOWED_HOSTS
    http_security_headers: Mapping[str, str] = field(
        default_factory=_default_studio_http_security_headers
    )
    request_id_header: str = "x-request-id"
    max_request_body_bytes: int = DEFAULT_STUDIO_MAX_REQUEST_BODY_BYTES
    enforce_route_policies: bool = False
    identity_file_path: str | None = None
    allow_header_principal: bool = True
    job_root_path: str | None = None
    job_default_timeout_seconds: float = DEFAULT_STUDIO_JOB_TIMEOUT_SECONDS
    audit_log_path: str | None = None
    audit_rotation_bytes: int | None = None
    audit_retained_files: int = DEFAULT_STUDIO_AUDIT_RETAINED_FILES

    def __post_init__(self) -> None:
        """Validate settings that affect Studio security boundaries."""

        if self.deployment_profile not in ("development", "production"):
            raise ValueError("Studio deployment profile must be development or production.")
        if not self.cors_allowed_origins:
            raise ValueError("Studio CORS origins must not be empty.")
        if any(origin == "*" for origin in self.cors_allowed_origins):
            raise ValueError("Studio runtime settings reject wildcard CORS origins.")
        if not self.websocket_allowed_origins:
            raise ValueError("Studio WebSocket origins must not be empty.")
        if any(origin == "*" for origin in self.websocket_allowed_origins):
            raise ValueError("Studio runtime settings reject wildcard WebSocket origins.")
        if not self.allowed_hosts:
            raise ValueError("Studio allowed hosts must not be empty.")
        if any(host == "*" for host in self.allowed_hosts):
            raise ValueError("Studio runtime settings reject wildcard hosts.")
        if any(not name.strip() for name in self.http_security_headers):
            raise ValueError("Studio security header names must not be empty.")
        if any(not value.strip() for value in self.http_security_headers.values()):
            raise ValueError("Studio security header values must not be empty.")
        if not self.request_id_header.strip():
            raise ValueError("Studio request ID header must not be empty.")
        if self.max_request_body_bytes <= 0:
            raise ValueError("Studio request body limit must be positive.")
        if self.identity_file_path is not None and not self.identity_file_path.strip():
            raise ValueError("Studio identity file path must not be empty.")
        if not isinstance(self.allow_header_principal, bool):
            raise ValueError("Studio header principal fallback must be boolean.")
        if self.job_root_path is not None and not self.job_root_path.strip():
            raise ValueError("Studio job root path must not be empty.")
        if self.job_default_timeout_seconds <= 0:
            raise ValueError("Studio job timeout must be positive.")
        if self.audit_log_path is not None and not self.audit_log_path.strip():
            raise ValueError("Studio audit log path must not be empty.")
        if self.audit_rotation_bytes is not None and self.audit_rotation_bytes <= 0:
            raise ValueError("Studio audit rotation byte limit must be positive.")
        if self.audit_retained_files < 0:
            raise ValueError("Studio retained audit file count must not be negative.")
        if self.deployment_profile == "production":
            self._validate_production_profile()

    def _validate_production_profile(self) -> None:
        """Validate fail-closed settings required for production deployments."""

        if not self.enforce_route_policies:
            raise ValueError("Studio production profile requires route policy enforcement.")
        if self.allow_header_principal:
            raise ValueError("Studio production profile requires header principal fallback disabled.")
        if self.identity_file_path is None:
            raise ValueError("Studio production profile requires an identity file.")
        if self.audit_log_path is None:
            raise ValueError("Studio production profile requires a persistent audit log.")
        if self.job_root_path is None:
            raise ValueError("Studio production profile requires a persistent job root.")


def build_default_studio_runtime_settings(
    env: Mapping[str, str] | None = None,
) -> StudioRuntimeSettings:
    """Build Studio runtime settings from environment-style values."""

    source = os.environ if env is None else env
    raw_deployment_profile = source.get("SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE")
    raw_origins = source.get("SC_NEUROCORE_STUDIO_CORS_ORIGINS")
    raw_websocket_origins = source.get("SC_NEUROCORE_STUDIO_WEBSOCKET_ALLOWED_ORIGINS")
    raw_hosts = source.get("SC_NEUROCORE_STUDIO_ALLOWED_HOSTS")
    raw_max_request_body_bytes = source.get("SC_NEUROCORE_STUDIO_MAX_REQUEST_BODY_BYTES")
    raw_enforce_route_policies = source.get("SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES")
    raw_identity_file_path = source.get("SC_NEUROCORE_STUDIO_IDENTITY_FILE")
    raw_allow_header_principal = source.get("SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL")
    raw_job_root_path = source.get("SC_NEUROCORE_STUDIO_JOB_ROOT")
    raw_job_default_timeout_seconds = source.get("SC_NEUROCORE_STUDIO_JOB_TIMEOUT_SECONDS")
    raw_audit_log_path = source.get("SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH")
    raw_audit_rotation_bytes = source.get("SC_NEUROCORE_STUDIO_AUDIT_ROTATION_BYTES")
    raw_audit_retained_files = source.get("SC_NEUROCORE_STUDIO_AUDIT_RETAINED_FILES")
    deployment_profile = _parse_deployment_profile(raw_deployment_profile)
    origins = (
        DEFAULT_STUDIO_CORS_ORIGINS
        if raw_origins is None or not raw_origins.strip()
        else tuple(origin.strip() for origin in raw_origins.split(",") if origin.strip())
    )
    hosts = (
        DEFAULT_STUDIO_ALLOWED_HOSTS
        if raw_hosts is None or not raw_hosts.strip()
        else tuple(host.strip() for host in raw_hosts.split(",") if host.strip())
    )
    websocket_origins = (
        origins
        if raw_websocket_origins is None or not raw_websocket_origins.strip()
        else tuple(origin.strip() for origin in raw_websocket_origins.split(",") if origin.strip())
    )
    try:
        max_request_body_bytes = (
            DEFAULT_STUDIO_MAX_REQUEST_BODY_BYTES
            if raw_max_request_body_bytes is None or not raw_max_request_body_bytes.strip()
            else int(raw_max_request_body_bytes)
        )
    except ValueError as exc:
        raise ValueError("Studio request body limit must be an integer.") from exc
    enforce_route_policies = _parse_bool_env(
        raw_enforce_route_policies,
        default=False,
        error_message="Studio route policy enforcement must be a boolean flag.",
    )
    allow_header_principal = _parse_bool_env(
        raw_allow_header_principal,
        default=True,
        error_message="Studio header principal fallback must be a boolean flag.",
    )
    identity_file_path = (
        None
        if raw_identity_file_path is None or not raw_identity_file_path.strip()
        else raw_identity_file_path.strip()
    )
    job_root_path = (
        None if raw_job_root_path is None or not raw_job_root_path.strip() else raw_job_root_path.strip()
    )
    try:
        job_default_timeout_seconds = (
            DEFAULT_STUDIO_JOB_TIMEOUT_SECONDS
            if raw_job_default_timeout_seconds is None
            or not raw_job_default_timeout_seconds.strip()
            else float(raw_job_default_timeout_seconds)
        )
    except ValueError as exc:
        raise ValueError("Studio job timeout must be numeric.") from exc
    audit_log_path = (
        None
        if raw_audit_log_path is None or not raw_audit_log_path.strip()
        else raw_audit_log_path.strip()
    )
    try:
        audit_rotation_bytes = (
            None
            if raw_audit_rotation_bytes is None or not raw_audit_rotation_bytes.strip()
            else int(raw_audit_rotation_bytes)
        )
    except ValueError as exc:
        raise ValueError("Studio audit rotation byte limit must be an integer.") from exc
    try:
        audit_retained_files = (
            DEFAULT_STUDIO_AUDIT_RETAINED_FILES
            if raw_audit_retained_files is None or not raw_audit_retained_files.strip()
            else int(raw_audit_retained_files)
        )
    except ValueError as exc:
        raise ValueError("Studio retained audit file count must be an integer.") from exc
    return StudioRuntimeSettings(
        deployment_profile=deployment_profile,
        cors_allowed_origins=origins,
        websocket_allowed_origins=websocket_origins,
        allowed_hosts=hosts,
        max_request_body_bytes=max_request_body_bytes,
        enforce_route_policies=enforce_route_policies,
        identity_file_path=identity_file_path,
        allow_header_principal=allow_header_principal,
        job_root_path=job_root_path,
        job_default_timeout_seconds=job_default_timeout_seconds,
        audit_log_path=audit_log_path,
        audit_rotation_bytes=audit_rotation_bytes,
        audit_retained_files=audit_retained_files,
    )


def _parse_deployment_profile(raw_value: str | None) -> StudioDeploymentProfile:
    if raw_value is None or not raw_value.strip():
        return "development"
    normalized = raw_value.strip().lower()
    if normalized == "development":
        return "development"
    if normalized == "production":
        return "production"
    raise ValueError("Studio deployment profile must be development or production.")


def _parse_bool_env(
    raw_value: str | None,
    *,
    default: bool,
    error_message: str,
) -> bool:
    if raw_value is None or not raw_value.strip():
        return default
    normalized = raw_value.strip().lower()
    if normalized in ("0", "false", "no"):
        return False
    if normalized in ("1", "true", "yes"):
        return True
    raise ValueError(error_message)
