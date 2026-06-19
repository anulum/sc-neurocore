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
from dataclasses import dataclass
from types import MappingProxyType

DEFAULT_STUDIO_CORS_ORIGINS: tuple[str, ...] = (
    "http://127.0.0.1:8001",
    "http://localhost:8001",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
)

DEFAULT_STUDIO_ALLOWED_HOSTS: tuple[str, ...] = (
    "127.0.0.1",
    "localhost",
)

DEFAULT_STUDIO_HTTP_SECURITY_HEADERS: Mapping[str, str] = MappingProxyType(
    {
        "x-content-type-options": "nosniff",
        "referrer-policy": "no-referrer",
        "x-frame-options": "DENY",
    }
)


@dataclass(frozen=True, slots=True)
class StudioRuntimeSettings:
    """Runtime settings consumed by the Studio FastAPI application."""

    cors_allowed_origins: tuple[str, ...] = DEFAULT_STUDIO_CORS_ORIGINS
    allowed_hosts: tuple[str, ...] = DEFAULT_STUDIO_ALLOWED_HOSTS
    http_security_headers: Mapping[str, str] = DEFAULT_STUDIO_HTTP_SECURITY_HEADERS
    request_id_header: str = "x-request-id"

    def __post_init__(self) -> None:
        """Validate settings that affect Studio security boundaries."""

        if not self.cors_allowed_origins:
            raise ValueError("Studio CORS origins must not be empty.")
        if any(origin == "*" for origin in self.cors_allowed_origins):
            raise ValueError("Studio runtime settings reject wildcard CORS origins.")
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


def build_default_studio_runtime_settings(
    env: Mapping[str, str] | None = None,
) -> StudioRuntimeSettings:
    """Build Studio runtime settings from environment-style values."""

    source = os.environ if env is None else env
    raw_origins = source.get("SC_NEUROCORE_STUDIO_CORS_ORIGINS")
    raw_hosts = source.get("SC_NEUROCORE_STUDIO_ALLOWED_HOSTS")
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
    return StudioRuntimeSettings(cors_allowed_origins=origins, allowed_hosts=hosts)
