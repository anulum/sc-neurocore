# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Graph custody across real backend processes

"""Recover graph identity and complete artifacts after a real backend process exits."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx

from tests.test_studio_graph_jobs import graph


@contextmanager
def server(root: Path, name: str) -> Iterator[tuple[httpx.Client, int]]:
    """Run one owned public ASGI module on an exclusively bound loopback socket."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("SC_NEUROCORE_STUDIO_")
        }
        environment.update(
            {
                "SC_NEUROCORE_STUDIO_DEPLOYMENT_PROFILE": "development",
                "SC_NEUROCORE_STUDIO_ENFORCE_ROUTE_POLICIES": "true",
                "SC_NEUROCORE_STUDIO_ALLOW_HEADER_PRINCIPAL": "true",
                "SC_NEUROCORE_STUDIO_PROJECT_ROOT": str(root / "projects"),
                "SC_NEUROCORE_STUDIO_JOB_ROOT": str(root / "jobs"),
                "SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH": str(root / "audit.jsonl"),
                "SC_NEUROCORE_STUDIO_BENCHMARK_DATABANK": str(root / "databank.jsonl"),
            }
        )
        command = [
            sys.executable,
            "-m",
            "uvicorn",
            "sc_neurocore.studio.app:app",
            "--fd",
            str(listener.fileno()),
            "--log-level",
            "warning",
        ]
        with (
            (root / (name + ".stdout")).open("wb") as stdout,
            (root / (name + ".stderr")).open("wb") as stderr,
        ):
            process = subprocess.Popen(
                command,
                env=environment,
                pass_fds=(listener.fileno(),),
                stdout=stdout,
                stderr=stderr,
            )
            try:
                with httpx.Client(
                    base_url=f"http://127.0.0.1:{port}",
                    timeout=5.0,
                    trust_env=False,
                    headers={"x-studio-principal": "alice", "sec-fetch-site": "none"},
                ) as client:
                    deadline = time.monotonic() + 40.0
                    while True:
                        assert process.poll() is None, "owned backend exited before readiness"
                        try:
                            if client.get("/api/health").status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        assert time.monotonic() < deadline, "owned backend did not become ready"
                        time.sleep(0.05)
                    yield client, process.pid
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=15.0)
                assert process.returncode in (0, -signal.SIGTERM)


def completed_graph(client: httpx.Client, route: str) -> dict[str, Any]:
    """Wait for the real HTTP worker to complete without inventing terminal state."""
    deadline = time.monotonic() + 90.0
    while True:
        response = client.get(route)
        assert response.status_code == 200, response.text
        record = response.json()
        if record["status"] == "completed":
            return dict(record)
        assert record["status"] in {"pending", "running"}, record
        assert time.monotonic() < deadline, "real graph worker did not complete"
        time.sleep(0.05)


def test_real_backend_process_restart_recovers_exact_graph_result_and_retry(tmp_path: Path) -> None:
    """A different server PID recovers one actor-owned job and its complete sealed bytes."""
    request = graph()
    retry = {"idempotency-key": "real-process-restart"}
    with server(tmp_path, "first") as (first, first_pid):
        admitted = first.post("/api/graph/jobs", json=request, headers=retry)
        assert admitted.status_code == 200, admitted.text
        receipt = admitted.json()
        record = completed_graph(first, receipt["status_route"])
        assert record["owner"] == "alice"
        result = first.get(receipt["status_route"] + "/result")
        assert result.status_code == 200, result.text
        original_bytes = result.content
        digest = hashlib.sha256(original_bytes).hexdigest()
        assert record["result"]["result_sha256"] == digest
    with server(tmp_path, "second") as (second, second_pid):
        assert second_pid != first_pid
        observed = second.get(receipt["status_route"])
        assert observed.status_code == 200 and observed.json() == record
        recovered = second.get(receipt["status_route"] + "/result")
        assert recovered.status_code == 200 and recovered.content == original_bytes
        assert recovered.headers["x-studio-artifact-sha256"] == digest
        replay = second.post("/api/graph/jobs", json=request, headers=retry)
        assert replay.status_code == 200 and replay.json()["job_id"] == record["job_id"]
        assert len(second.get("/api/graph/jobs").json()["jobs"]) == 1
        for suffix in ("", "/result"):
            refused = second.get(
                receipt["status_route"] + suffix, headers={"x-studio-principal": "bob"}
            )
            assert refused.status_code == 404
    (tmp_path / "process-restart-proof.json").write_text(
        json.dumps(
            {
                "schema_version": "graph-process-restart-proof.v1",
                "first_pid": first_pid,
                "second_pid": second_pid,
                "job_id": record["job_id"],
                "result_sha256": digest,
                "result_bytes": len(original_bytes),
                "both_owned_servers_reaped": True,
                "profile": "development header identity; functional source backend restart, no installed or desktop claim",
            },
            indent=2,
        )
    )
