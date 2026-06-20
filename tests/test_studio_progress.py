# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio WebSocket progress streaming

from __future__ import annotations

import hashlib
import json
import queue
from pathlib import Path
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import Principal, StudioRuntimeSettings
from sc_neurocore.studio.progress import (
    _characterize_with_progress,
    _heatmap_with_progress,
    _scan_with_progress,
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(
        create_app(runtime_settings=StudioRuntimeSettings(allowed_hosts=("testserver",))),
        headers={"origin": "http://127.0.0.1:8001"},
    )


# --- Characterise with progress ---


class TestCharacteriseProgress:
    def test_produces_progress_and_complete(self) -> None:
        q: queue.Queue[dict[str, Any]] = queue.Queue()

        def sim_fn(**kw: Any) -> dict[str, Any]:
            import numpy as np

            n = 100
            return {
                "time": list(range(n)),
                "states": {"v": list(np.zeros(n))},
                "spikes": [],
                "spike_count": 0,
                "stats": {"rate_hz": 0, "isi_mean_ms": None, "isi_cv": None, "isi_histogram": None},
                "dt": 0.1,
                "n_steps": n,
                "current_trace": list(np.zeros(n)),
            }

        _characterize_with_progress(sim_fn, {"current": 10.0, "params": {"a": 1.0}}, q)

        messages = []
        while not q.empty():
            messages.append(q.get_nowait())

        types = [m["type"] for m in messages]
        assert "progress" in types
        assert types[-1] == "complete"

        complete = messages[-1]
        assert "result" in complete
        assert "fi_curve" in complete["result"]
        assert "pattern" in complete["result"]

    def test_progress_has_pct(self) -> None:
        q: queue.Queue[dict[str, Any]] = queue.Queue()

        def sim_fn(**kw: Any) -> dict[str, Any]:
            import numpy as np

            n = 50
            return {
                "time": list(range(n)),
                "states": {"v": list(np.zeros(n))},
                "spikes": [],
                "spike_count": 0,
                "stats": {"rate_hz": 0, "isi_mean_ms": None, "isi_cv": None, "isi_histogram": None},
                "dt": 0.1,
                "n_steps": n,
                "current_trace": list(np.zeros(n)),
            }

        _characterize_with_progress(sim_fn, {"current": 5.0}, q)

        progress_msgs = []
        while not q.empty():
            m = q.get_nowait()
            if m["type"] == "progress":
                progress_msgs.append(m)

        assert len(progress_msgs) > 0
        for m in progress_msgs:
            assert "pct" in m
            assert 0 <= m["pct"] <= 100
            assert "msg" in m


# --- Heatmap with progress ---


class TestHeatmapProgress:
    def test_heatmap_progress(self) -> None:
        q: queue.Queue[dict[str, Any]] = queue.Queue()

        def sim_fn(**kw: Any) -> dict[str, Any]:
            import numpy as np

            n = 50
            return {
                "time": list(range(n)),
                "states": {"v": list(np.zeros(n))},
                "spikes": [],
                "spike_count": 0,
                "stats": {"rate_hz": 0, "isi_mean_ms": None, "isi_cv": None, "isi_histogram": None},
                "dt": 0.1,
                "n_steps": n,
                "current_trace": list(np.zeros(n)),
            }

        _heatmap_with_progress(
            sim_fn,
            {"params": {"a": 1.0}, "current": 10.0},
            "a",
            [0.5, 1.0, 1.5],
            "a",
            [0.5, 1.0, 1.5],
            q,
        )

        messages = []
        while not q.empty():
            messages.append(q.get_nowait())

        types = [m["type"] for m in messages]
        assert "progress" in types
        assert types[-1] == "complete"

        complete = messages[-1]
        assert "rates" in complete["result"]
        assert len(complete["result"]["rates"]) == 3


# --- Scan with progress ---


class TestScanProgress:
    def test_scan_produces_results(self) -> None:
        q: queue.Queue[dict[str, Any]] = queue.Queue()
        _scan_with_progress(q)

        messages = []
        while not q.empty():
            messages.append(q.get_nowait())

        types = [m["type"] for m in messages]
        assert "progress" in types
        assert types[-1] == "complete"

        complete = messages[-1]
        assert isinstance(complete["result"], list)
        assert len(complete["result"]) > 0
        assert "name" in complete["result"][0]


# --- WebSocket endpoint ---


class TestWebSocketEndpoint:
    def test_ws_unknown_op(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/progress") as ws:
            ws.send_json({"op": "nonexistent"})
            msg = ws.receive_json()
            assert msg["type"] == "error"
            assert "Unknown op" in msg["msg"]

    def test_ws_invalid_json(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/progress") as ws:
            ws.send_text("not json at all")
            msg = ws.receive_json()
            assert msg["type"] == "error"

    def test_ws_characterize_streams_progress(self, client: TestClient) -> None:
        with client.websocket_connect("/ws/progress") as ws:
            ws.send_json(
                {
                    "op": "characterize",
                    "config": {"name": "LIFNeuron", "current": 10.0, "duration": 50.0},
                }
            )
            messages = []
            for _ in range(100):
                msg = ws.receive_json()
                messages.append(msg)
                if msg["type"] in ("complete", "error"):
                    break

            types = [m["type"] for m in messages]
            assert "complete" in types or "error" in types
            if "complete" in types:
                progress_count = types.count("progress")
                assert progress_count > 0

    def test_ws_rejects_unconfigured_origin(self) -> None:
        app = create_app(
            runtime_settings=StudioRuntimeSettings(
                allowed_hosts=("testserver",),
                websocket_allowed_origins=("https://studio.example.test",),
            )
        )
        client = TestClient(app)

        with (
            pytest.raises(WebSocketDisconnect) as exc_info,
            client.websocket_connect(
                "/ws/progress",
                headers={"origin": "https://attacker.example.test"},
            ),
        ):
            pass

        assert exc_info.value.code == 1008

    def test_ws_policy_rejects_missing_principal(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit" / "studio.jsonl"
        app = create_app(
            runtime_settings=StudioRuntimeSettings(
                allowed_hosts=("testserver",),
                audit_log_path=str(audit_path),
                enforce_route_policies=True,
            )
        )
        client = TestClient(app)

        with (
            pytest.raises(WebSocketDisconnect) as exc_info,
            client.websocket_connect(
                "/ws/progress",
                headers={"origin": "http://127.0.0.1:8001"},
            ),
        ):
            pass

        row = json.loads(audit_path.read_text(encoding="utf-8"))
        assert exc_info.value.code == 1008
        assert row["action"] == "studio.websocket.progress"
        assert row["decision"] == "deny"
        assert row["principal_id"] is None
        assert row["reason"] == "missing_principal"
        assert row["route"] == "/ws/progress"

    def test_ws_policy_rejects_invalid_bearer_identity_token(self, tmp_path: Path) -> None:
        identity_path = tmp_path / "studio-identities.json"
        audit_path = tmp_path / "audit" / "studio.jsonl"
        identity_path.write_text(
            json.dumps(
                {
                    "schema_version": "sc-neurocore.studio.identity.v1",
                    "service_accounts": [
                        {
                            "principal_id": "svc-admin",
                            "roles": ["studio.admin"],
                            "token_sha256": hashlib.sha256(b"admin-token").hexdigest(),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        app = create_app(
            runtime_settings=StudioRuntimeSettings(
                allowed_hosts=("testserver",),
                allow_header_principal=False,
                audit_log_path=str(audit_path),
                enforce_route_policies=True,
                identity_file_path=str(identity_path),
            )
        )
        client = TestClient(app)

        with (
            pytest.raises(WebSocketDisconnect) as exc_info,
            client.websocket_connect(
                "/ws/progress",
                headers={
                    "authorization": "Bearer wrong-token",
                    "origin": "http://127.0.0.1:8001",
                },
            ),
        ):
            pass

        row = json.loads(audit_path.read_text(encoding="utf-8"))
        assert exc_info.value.code == 1008
        assert row["action"] == "studio.websocket.progress"
        assert row["decision"] == "deny"
        assert row["reason"] == "invalid_identity_token"

    def test_ws_policy_accepts_browser_session_subprotocol(self, tmp_path: Path) -> None:
        audit_path = tmp_path / "audit" / "studio.jsonl"
        app = create_app(
            runtime_settings=StudioRuntimeSettings(
                allowed_hosts=("testserver",),
                allow_header_principal=False,
                audit_log_path=str(audit_path),
                enforce_route_policies=True,
            )
        )
        issued = app.state.studio_browser_session_manager.issue(
            Principal(principal_id="browser-operator", roles=frozenset({"studio.viewer"}))
        )
        client = TestClient(app)

        with client.websocket_connect(
            "/ws/progress",
            headers={"origin": "http://127.0.0.1:8001"},
            subprotocols=["studio-auth", f"studio-bearer.{issued.bearer_token}"],
        ) as ws:
            ws.send_json({"op": "nonexistent"})
            message = ws.receive_json()

        row = json.loads(audit_path.read_text(encoding="utf-8"))
        assert message["type"] == "error"
        assert "Unknown op" in message["msg"]
        assert row["action"] == "studio.websocket.progress"
        assert row["decision"] == "allow"
        assert row["principal_id"] == "browser-operator"
        assert row["reason"] == "authorized"
