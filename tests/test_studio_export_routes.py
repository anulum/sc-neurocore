# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio export route tests

"""Exercise SVG export and progress WebSocket adapter branches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import WebSocket
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from sc_neurocore.studio.api import export as export_routes
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import (
    AuditSinkError,
    PolicyGateway,
    StudioRuntimeSettings,
)

_ORIGIN = "http://127.0.0.1:8001"


def _client(tmp_path: Path, *, enforce_route_policies: bool = False) -> TestClient:
    """Return an isolated client that admits the canonical local origin."""

    application = create_app(
        StudioRuntimeSettings(
            allowed_hosts=("testserver",),
            audit_log_path=str(tmp_path / "audit" / "studio.jsonl"),
            enforce_route_policies=enforce_route_policies,
            job_root_path=str(tmp_path / "jobs"),
            websocket_allowed_origins=(_ORIGIN,),
        )
    )
    return TestClient(application)


def test_svg_export_returns_rendered_vector_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HTTP adapter simulates once and returns SVG media."""

    def _simulate_model(**_kwargs: object) -> dict[str, Any]:
        return {
            "model_name": "LIFNeuron",
            "spikes": [1],
            "states": {"v": [-65.0, -50.0, -65.0]},
            "time": [0.0, 0.1, 0.2],
        }

    monkeypatch.setattr(export_routes, "simulate_model", _simulate_model)
    response = _client(tmp_path).post(
        "/api/export/svg",
        json={"current": 1.0, "dt": 0.1, "duration": 0.3, "name": "LIFNeuron"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/svg+xml"
    assert response.text.startswith("<svg")
    assert "LIFNeuron" in response.text


def test_progress_websocket_closes_when_policy_audit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An audit outage closes the policy-gated socket with an internal code."""

    def _raise_audit_error(self: PolicyGateway, *_args: object, **_kwargs: object) -> object:
        del self
        raise AuditSinkError("private/audit/path")

    monkeypatch.setattr(PolicyGateway, "authorize", _raise_audit_error)
    client = _client(tmp_path, enforce_route_policies=True)

    with (
        pytest.raises(WebSocketDisconnect) as exc_info,
        client.websocket_connect("/ws/progress", headers={"origin": _ORIGIN}),
    ):
        pass

    assert exc_info.value.code == 1011


def test_progress_websocket_treats_peer_disconnect_as_normal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disconnect raised by the stream handler is consumed after acceptance."""

    async def _disconnect(websocket: WebSocket) -> None:
        del websocket
        raise WebSocketDisconnect(code=1000)

    import sc_neurocore.studio.progress as progress

    monkeypatch.setattr(progress, "ws_progress_handler", _disconnect)
    client = _client(tmp_path)

    with client.websocket_connect("/ws/progress", headers={"origin": _ORIGIN}):
        pass
