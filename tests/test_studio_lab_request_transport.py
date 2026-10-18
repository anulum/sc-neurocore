# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Lab CLI and real HTTP transport acceptance

"""Exercise CLI startup, chunked uploads and identity restart over real loopback TCP."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest


def _port() -> int:
    """Allocate a free loopback port without reusing a peer server."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _cli(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run a finite real CLI command, keeping generated credentials out of logs."""
    return subprocess.run(
        [sys.executable, "-m", "sc_neurocore.cli", *args],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )


def _environment(root: Path) -> dict[str, str]:
    """Keep this server's state and configuration separate from its parent session."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SC_NEUROCORE_STUDIO_")
    }
    env.update(
        {
            "BROWSER": "/bin/true",
            "SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE": "production",
            "SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES": "true",
            "SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL": "false",
            "SC_NEUROCORE_STUDIO_IDENTITY_FILE": str(root / "identities.json"),
            "SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH": str(root / "audit.jsonl"),
            "SC_NEUROCORE_STUDIO_JOB_ROOT": str(root / "jobs"),
            "SC_NEUROCORE_STUDIO_MAX_REQUEST_BODY_BYTES": "128",
        }
    )
    return env


@contextmanager
def _server(
    port: int, env: dict[str, str], *, module_entry: bool = False
) -> Iterator[httpx.Client]:
    """Start exactly one owned CLI server and stop that process after acceptance."""
    command = [sys.executable, "-m", "sc_neurocore.cli", "studio", "--port", str(port)]
    if module_entry:
        # Exercise a preloading ASGI launcher and its subsequent public string lookup.
        command = [
            sys.executable,
            "-c",
            "import sys, uvicorn; from sc_neurocore.studio.app import app; "
            "uvicorn.run('sc_neurocore.studio.app:app', host='127.0.0.1', "
            "port=int(sys.argv[1]), log_level='warning')",
            str(port),
        ]
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}",
            headers={"origin": f"http://127.0.0.1:{port}"},
            timeout=5,
        ) as client:
            deadline = time.monotonic() + 25
            while True:
                assert process.poll() is None, "Studio exited before readiness"
                try:
                    if client.get("/api/health").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                assert time.monotonic() < deadline, "Studio did not become ready"
                time.sleep(0.05)
            yield client
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=10)


def test_actual_chunked_upload_obeys_received_byte_budget(tmp_path: Path) -> None:
    """The real HTTP server refuses an over-budget chunked public login before verification."""
    env = _environment(tmp_path)
    bootstrap = _cli(
        "studio-bootstrap-admin", "--identity-file", str(tmp_path / "identities.json"), env=env
    )
    assert bootstrap.returncode == 0
    payload = json.dumps({"username": "operator", "password": "x" * 256}).encode()
    port = _port()
    env["SC_NEUROCORE_STUDIO_CORS_ORIGINS"] = f"http://127.0.0.1:{port}"
    env["SC_NEUROCORE_STUDIO_WEBSOCKET_ALLOWED_ORIGINS"] = f"http://127.0.0.1:{port}"
    with _server(port, env) as client:
        response = client.post(
            "/api/studio/auth/login",
            headers={"content-type": "application/json"},
            content=iter([payload[:64], payload[64:128], payload[128:]]),
        )
        assert response.request.headers["transfer-encoding"] == "chunked"
        assert "content-length" not in response.request.headers
        assert response.status_code == 413
        assert response.json() == {"detail": "Studio request body exceeds configured limit."}
        assert not (tmp_path / "audit.jsonl").exists()
        token = json.loads(bootstrap.stdout)["bearer_token"]
        jobs = client.get("/api/studio/jobs", headers={"authorization": f"Bearer {token}"})
        assert jobs.status_code == 200
        assert jobs.json()["jobs"] == []
        for route in ("/docs", "/redoc"):
            documentation = client.get(route)
            assert documentation.status_code == 200
            assert "'sha256-" in documentation.headers["content-security-policy"]
            assert (
                "script-src 'self' https://cdn.jsdelivr.net"
                in documentation.headers["content-security-policy"]
            )


def test_cli_identity_replacement_is_honoured_after_server_restart(tmp_path: Path) -> None:
    """A real CLI replacement revokes the old bearer at the documented restart boundary."""
    env = _environment(tmp_path)
    identity_args = ("studio-bootstrap-admin", "--identity-file", str(tmp_path / "identities.json"))
    initial = _cli(*identity_args, env=env)
    assert initial.returncode == 0
    old_token = json.loads(initial.stdout)["bearer_token"]
    port = _port()
    route = "/api/studio/identity/service-accounts"
    with _server(port, env) as client:
        assert (
            client.get(route, headers={"authorization": f"Bearer {old_token}"}).status_code == 200
        )
        replaced = _cli(*identity_args, "--allow-overwrite", env=env)
        assert replaced.returncode == 0
        assert "Restart a running Studio API" in replaced.stderr
        new_token = json.loads(replaced.stdout)["bearer_token"]
    with _server(port, env) as client:
        assert (
            client.get(route, headers={"authorization": f"Bearer {old_token}"}).status_code == 401
        )
        assert (
            client.get(route, headers={"authorization": f"Bearer {new_token}"}).status_code == 200
        )


@pytest.mark.parametrize(
    "variable,value",
    [
        ("SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES", "false"),
        ("SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL", "true"),
        ("SC_NEUROCORE_STUDIO_BENCHMARK_MAX_BYTES", "0"),
        ("SC_NEUROCORE_STUDIO_BENCHMARK_MAX_RECORDS", "0"),
        ("SC_NEUROCORE_STUDIO_BENCHMARK_DATABANK", " "),
        ("SC_NEUROCORE_STUDIO_BROWSER_LOGIN_MAX_BUCKETS", "0"),
    ],
)
def test_inconsistent_identity_settings_never_listen(
    tmp_path: Path, variable: str, value: str
) -> None:
    """Refuse inconsistent lab identities or resource budgets before listening."""
    env = _environment(tmp_path)
    assert (
        _cli(
            "studio-bootstrap-admin", "--identity-file", str(tmp_path / "identities.json"), env=env
        ).returncode
        == 0
    )
    env[variable] = value
    port = _port()
    result = _cli("studio", "--port", str(port), env=env)
    assert result.returncode == 1
    assert "Studio could not start" in result.stdout
    with socket.socket() as connection:
        assert connection.connect_ex(("127.0.0.1", port)) != 0


@pytest.mark.parametrize(
    "variable,value,port_override",
    [
        ("", "", 0),
        ("", "", 65536),
        ("SC_NEUROCORE_STUDIO_CORS_ORIGINS", "http://127.0.0.1:9", None),
        ("SC_NEUROCORE_STUDIO_WEBSOCKET_ALLOWED_ORIGINS", "http://127.0.0.1:9", None),
    ],
)
def test_invalid_launcher_port_or_origins_never_creates_a_runtime(
    tmp_path: Path, variable: str, value: str, port_override: int | None
) -> None:
    """Reject invalid ports and mismatched explicit origins through the actual CLI."""
    env = _environment(tmp_path)
    assert (
        _cli(
            "studio-bootstrap-admin", "--identity-file", str(tmp_path / "identities.json"), env=env
        ).returncode
        == 0
    )
    if variable:
        env[variable] = value
    port = _port() if port_override is None else port_override
    result = _cli("studio", "--port", str(port), env=env)
    assert result.returncode == 1
    assert "Studio could not start" in result.stdout
    assert not (tmp_path / "jobs").exists()


def test_preloaded_module_app_serves_the_authenticated_asgi_entry(tmp_path: Path) -> None:
    """Serve the public ASGI app after preload and repeated launcher lookup."""
    env = _environment(tmp_path)
    bootstrap = _cli(
        "studio-bootstrap-admin", "--identity-file", str(tmp_path / "identities.json"), env=env
    )
    assert bootstrap.returncode == 0
    token = json.loads(bootstrap.stdout)["bearer_token"]
    with _server(_port(), env, module_entry=True) as client:
        response = client.get("/api/studio/jobs", headers={"authorization": f"Bearer {token}"})
        assert response.status_code == 200
        assert response.json()["jobs"] == []
        assert client.get("/docs").status_code == 200
