# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio progress web socket endpoint

"""Focused suite: TestWebSocketEndpoint from former test_studio_progress.py."""

from __future__ import annotations

from tests.studio_progress_support import *  # noqa: F403

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

