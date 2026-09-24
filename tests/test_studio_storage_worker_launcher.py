# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — direct-spawn worker launcher

"""The real launcher service starts, replays and stops exact worker generations.

The launcher runs as its own process through its command-line entry, because
it makes itself a child subreaper and kills unattributed adopted processes.
These same-identity runs establish function only: an unprivileged launcher
starts workers under its own UID and never qualifies isolation. Distinct-UID
refusal and success belong to the disposable three-identity proof.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.storage_launcher_client import (
    exchange_launcher_request,
    new_launcher_request,
)
from sc_neurocore.studio.platform.storage_launcher_protocol import (
    LauncherOperation,
)
from sc_neurocore.studio.platform.storage_transport import write_frame
from sc_neurocore.studio.platform.storage_worker_spawn import compute_identity_processes
from sc_neurocore.studio.platform.storage_worker_grant import (
    GRANT_ENDPOINT_NAME,
    ExpectedWorker,
    WorkerGrantEndpoint,
)
from tests.studio_storage_launcher_support import *


@pytest.fixture
def base() -> Iterator[Path]:
    """Short private base directory so every socket path fits the Unix limit."""
    with launcher_base() as path:
        yield path


@pytest.fixture
def launcher(base: Path) -> Iterator[Launcher]:
    """A running same-identity launcher; stopped with SIGTERM afterwards."""
    running = start(base)
    try:
        yield running
    finally:
        shutdown(running)


def test_launched_generation_runs_the_named_task_after_grant(launcher: Launcher) -> None:
    """Launch, grant and completion of one exact generation through the real service."""
    directory = prepare(launcher.spool_root, JOB_A, GEN_A)
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    registered: list[str] = []
    try:
        with WorkerGrantEndpoint(descriptor, directory, GRANT_ENDPOINT_NAME) as endpoint:
            started = send(launcher, "launch", JOB_A, GEN_A)
            assert started.state == "running" and started.pid is not None
            assert started.start_token is not None
            endpoint.grant(
                ExpectedWorker(uid=os.getuid(), pid=started.pid, start_token=started.start_token),
                deadline=time.monotonic() + 60.0,
                max_refusals=1,
                register=registered.append,
            )
    finally:
        os.close(descriptor)
    finished = await_state(launcher, JOB_A, GEN_A, "stopped")
    assert (finished.state, finished.exit_status) == ("stopped", 1)
    assert (finished.pid, finished.start_token) == (started.pid, started.start_token)
    assert registered == [
        f"{supervisor_identity().split(':')[0]}:{started.pid}:{started.start_token}"
    ]
    evidence = json.loads((directory / JOB_A / ".studio_process_result.json").read_text())
    assert evidence["error"] == "AnalysisJobValidationError"


def test_retry_and_status_resolve_a_lost_launch_reply(launcher: Launcher) -> None:
    """Retrying the same generation returns the recorded worker, never a second one."""
    prepare(launcher.spool_root, JOB_A, GEN_A)
    first = send(launcher, "launch", JOB_A, GEN_A)
    retried = send(launcher, "launch", JOB_A, GEN_A)
    status = send(launcher, "status", JOB_A, GEN_A)
    assert first.state == "running"
    for response in (retried, status):
        assert (response.pid, response.start_token) == (first.pid, first.start_token)
    other = send(launcher, "launch", JOB_A, GEN_B)
    assert (other.state, other.reason) == ("refused", "conflict")
    stopped = send(launcher, "stop", JOB_A, GEN_A)
    assert (stopped.state, stopped.pid, stopped.exit_status) == ("stopped", first.pid, -9)
    repeated = send(launcher, "stop", JOB_A, GEN_A)
    assert (repeated.state, repeated.exit_status) == ("stopped", -9)


def test_unknown_generation_is_absent(launcher: Launcher) -> None:
    """Status and stop for a generation the launcher never started report absence."""
    operations: tuple[LauncherOperation, ...] = ("status", "stop")
    for operation in operations:
        response = send(launcher, operation, JOB_B, GEN_B)
        assert response.state == "absent" and response.pid is None


def test_missing_spool_refuses_launch(launcher: Launcher) -> None:
    """No API-prepared generation directory means no worker is started."""
    response = send(launcher, "launch", JOB_B, GEN_B)
    assert (response.state, response.reason) == ("refused", "spool")
    assert send(launcher, "status", JOB_B, GEN_B).state == "absent"


def test_stop_kills_a_live_worker_tree(launcher: Launcher) -> None:
    """A worker waiting for its grant is stopped and confirmed gone."""
    prepare(launcher.spool_root, JOB_A, GEN_A)
    started = send(launcher, "launch", JOB_A, GEN_A)
    assert started.state == "running" and started.pid is not None
    stopped = send(launcher, "stop", JOB_A, GEN_A)
    assert (stopped.state, stopped.pid) == ("stopped", started.pid)
    assert not Path(f"/proc/{started.pid}").exists()
    assert send(launcher, "status", JOB_A, GEN_A).state == "stopped"


def test_worker_capacity_is_enforced(base: Path) -> None:
    """With one worker slot a second live generation is refused."""
    running = start(base, max_workers=1)
    try:
        prepare(running.spool_root, JOB_A, GEN_A)
        prepare(running.spool_root, JOB_B, GEN_B)
        assert send(running, "launch", JOB_A, GEN_A).state == "running"
        refused = send(running, "launch", JOB_B, GEN_B)
        assert (refused.state, refused.reason) == ("refused", "capacity")
        assert send(running, "stop", JOB_A, GEN_A).state == "stopped"
        admitted = send(running, "launch", JOB_B, GEN_B)
        assert admitted.state == "running"
        assert send(running, "stop", JOB_B, GEN_B).state == "stopped"
    finally:
        shutdown(running)


def test_record_budget_retires_only_stopped_generations(base: Path) -> None:
    """The oldest stopped record makes room; a live one never does."""
    running = start(base, max_records=1)
    try:
        prepare(running.spool_root, JOB_A, GEN_A)
        prepare(running.spool_root, JOB_B, GEN_B)
        assert send(running, "launch", JOB_A, GEN_A).state == "running"
        refused = send(running, "launch", JOB_B, GEN_B)
        assert (refused.state, refused.reason) == ("refused", "capacity")
        assert send(running, "stop", JOB_A, GEN_A).state == "stopped"
        assert send(running, "launch", JOB_B, GEN_B).state == "running"
        assert send(running, "status", JOB_A, GEN_A).state == "absent"
        assert send(running, "stop", JOB_B, GEN_B).state == "stopped"
    finally:
        shutdown(running)


def test_unstartable_bootstrap_is_reported_unavailable(base: Path) -> None:
    """A spawn failure is a refusal, not an invented worker identity."""
    running = start(base, python_executable=str(base / "missing-python"))
    try:
        prepare(running.spool_root, JOB_A, GEN_A)
        refused = send(running, "launch", JOB_A, GEN_A)
        assert (refused.state, refused.reason, refused.pid) == ("refused", "unavailable", None)
    finally:
        shutdown(running)


def test_compute_identity_scan_reports_live_processes_of_that_identity() -> None:
    """The start-up scan finds this identity's processes and nobody else's."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        mine = compute_identity_processes(os.getuid())
        assert {os.getpid(), child.pid} <= set(mine)
        foreign = compute_identity_processes(os.getuid() + 1_000_000)
        assert not {os.getpid(), child.pid} & set(foreign)
    finally:
        child.kill()
        child.wait(timeout=10.0)


def test_non_api_peer_and_malformed_frames_get_no_reply(base: Path) -> None:
    """Refused peers and malformed requests are closed; the service keeps serving."""
    running = start(base, api_uid=os.getuid() + 1)
    try:
        request = new_launcher_request("status", job_id=JOB_A, generation=GEN_A)
        with pytest.raises((EOFError, ConnectionError, OSError)):
            exchange_launcher_request(
                running.socket_path,
                request,
                launcher_uid=os.getuid(),
                deadline=time.monotonic() + 10.0,
            )
    finally:
        shutdown(running)
    served = start(base)
    try:
        for payload in (b"{}", b"not json"):
            channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            with channel:
                channel.connect(str(served.socket_path))
                write_frame(channel, payload, max_bytes=1024, deadline=time.monotonic() + 5.0)
                channel.settimeout(10.0)
                assert channel.recv(16) == b""
        assert send(served, "status", JOB_A, GEN_A).state == "absent"
    finally:
        shutdown(served)


def test_client_refuses_endpoint_not_owned_by_the_launcher(launcher: Launcher) -> None:
    """The API sends nothing to an endpoint owned by another identity."""
    request = new_launcher_request("status", job_id=JOB_A, generation=GEN_A)
    with pytest.raises(PermissionError):
        exchange_launcher_request(
            launcher.socket_path,
            request,
            launcher_uid=os.getuid() + 1,
            deadline=time.monotonic() + 10.0,
        )
    with pytest.raises(TimeoutError, match="already expired"):
        exchange_launcher_request(
            launcher.socket_path, request, launcher_uid=os.getuid(), deadline=time.monotonic()
        )
    with pytest.raises(ValidationError):
        new_launcher_request("status", job_id="sj_bad", generation=GEN_A)
