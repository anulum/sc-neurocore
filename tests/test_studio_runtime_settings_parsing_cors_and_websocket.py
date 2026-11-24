# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (cors_and_websocket) from former test_studio_runtime_settings_parsing.py

"""Parsing the Studio runtime settings: CORS and WebSocket origins."""

from __future__ import annotations

import pytest
from tests.studio_settings_support import (
    StudioRuntimeSettings,
    build_default_studio_runtime_settings,
)


def test_studio_runtime_settings_default_cors_origins_are_loopback_only() -> None:
    """The default CORS origins are loopback addresses only."""
    settings = build_default_studio_runtime_settings(env={})

    assert "http://127.0.0.1:8001" in settings.cors_allowed_origins
    assert "http://localhost:5173" in settings.cors_allowed_origins
    assert "*" not in settings.cors_allowed_origins


def test_studio_runtime_settings_parses_comma_separated_cors_origins() -> None:
    """Comma-separated CORS origins are parsed in order."""
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


def test_studio_runtime_settings_default_websocket_origins_match_cors() -> None:
    """The default WebSocket origins equal the CORS origins."""
    settings = build_default_studio_runtime_settings(env={})

    assert settings.websocket_allowed_origins == settings.cors_allowed_origins
    assert "*" not in settings.websocket_allowed_origins


def test_studio_runtime_settings_parses_comma_separated_websocket_origins() -> None:
    """Comma-separated WebSocket origins are parsed in order."""
    settings = build_default_studio_runtime_settings(
        env={
            "SC_NEUROCORE_STUDIO_WEBSOCKET_ALLOWED_ORIGINS": (
                "https://studio.example.test, http://127.0.0.1:9000 "
            )
        }
    )

    assert settings.websocket_allowed_origins == (
        "https://studio.example.test",
        "http://127.0.0.1:9000",
    )


def test_studio_runtime_settings_rejects_wildcard_websocket_origin() -> None:
    """A wildcard WebSocket origin is rejected with ValueError."""
    with pytest.raises(ValueError, match="wildcard WebSocket"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_WEBSOCKET_ALLOWED_ORIGINS": "http://localhost:5173,*"}
        )


def test_studio_runtime_settings_rejects_empty_websocket_origin_list() -> None:
    """An empty WebSocket origin list is rejected with ValueError."""
    with pytest.raises(ValueError, match="WebSocket origins"):
        StudioRuntimeSettings(websocket_allowed_origins=())


def test_studio_runtime_settings_rejects_wildcard_cors_origin() -> None:
    """A wildcard CORS origin is rejected with ValueError."""
    with pytest.raises(ValueError, match="wildcard CORS"):
        build_default_studio_runtime_settings(
            env={"SC_NEUROCORE_STUDIO_CORS_ORIGINS": "http://localhost:5173,*"}
        )


def test_studio_runtime_settings_rejects_empty_cors_origin_list() -> None:
    """An empty CORS origin list is rejected with ValueError."""
    with pytest.raises(ValueError, match="must not be empty"):
        StudioRuntimeSettings(cors_allowed_origins=())
