# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio runtime settings contract tests

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import (
    StudioRuntimeSettings,
    build_default_studio_runtime_settings,
)


def test_studio_runtime_settings_default_cors_origins_are_loopback_only() -> None:
    settings = build_default_studio_runtime_settings(env={})

    assert "http://127.0.0.1:8001" in settings.cors_allowed_origins
    assert "http://localhost:5173" in settings.cors_allowed_origins
    assert "*" not in settings.cors_allowed_origins


def test_studio_runtime_settings_parses_comma_separated_cors_origins() -> None:
    settings = build_default_studio_runtime_settings(
        env={
            "SC_NEUROCORE_STUDIO_CORS_ORIGINS": (
                "https://studio.example.test, http://127.0.0.1:9000 "
            )
        }
    )

    assert settings.cors_allowed_origins == (
        "https://studio.example.test",
        "http://127.0.0.1:9000",
    )


def test_studio_runtime_settings_rejects_wildcard_cors_origin() -> None:
    with pytest.raises(ValueError, match="wildcard CORS"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_CORS_ORIGINS": "http://localhost:5173,*"}
        )


def test_studio_runtime_settings_rejects_empty_cors_origin_list() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        StudioRuntimeSettings(cors_allowed_origins=())


def test_studio_runtime_settings_default_security_headers_are_fail_closed() -> None:
    settings = build_default_studio_runtime_settings(env={})

    assert settings.http_security_headers["x-content-type-options"] == "nosniff"
    assert settings.http_security_headers["referrer-policy"] == "no-referrer"
    assert settings.http_security_headers["x-frame-options"] == "DENY"


def test_studio_runtime_settings_rejects_empty_security_header_name() -> None:
    with pytest.raises(ValueError, match="security header names"):
        StudioRuntimeSettings(http_security_headers={"": "nosniff"})


def test_studio_runtime_settings_rejects_empty_security_header_value() -> None:
    with pytest.raises(ValueError, match="security header values"):
        StudioRuntimeSettings(http_security_headers={"x-content-type-options": ""})


def test_studio_app_adds_default_security_headers_to_health_response() -> None:
    client = TestClient(create_app())

    response = client.get("/api/health")

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["x-frame-options"] == "DENY"


def test_studio_app_cors_preflight_allows_configured_origin() -> None:
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            cors_allowed_origins=("https://studio.example.test",)
        )
    )
    client = TestClient(app)

    response = client.options(
        "/api/health",
        headers={
            "Origin": "https://studio.example.test",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://studio.example.test"


def test_studio_app_cors_preflight_rejects_unconfigured_origin() -> None:
    app = create_app(
        runtime_settings=StudioRuntimeSettings(
            cors_allowed_origins=("https://studio.example.test",)
        )
    )
    client = TestClient(app)

    response = client.options(
        "/api/health",
        headers={
            "Origin": "https://attacker.example.test",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers
