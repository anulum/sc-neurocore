# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — launcher-started worker grant with real processes

"""Real worker processes must earn the grant before any task import."""

from __future__ import annotations

import json
import os
import socket
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_reaper import reap_process_group
from sc_neurocore.studio.platform.storage_worker_grant import (
    ExpectedWorker,
    WorkerGrantEndpoint,
)
from tests.studio_storage_grant_support import *


@pytest.fixture
def grant_dir() -> Iterator[tuple[int, Path]]:
    """Hold a short private directory so the socket path fits the Unix limit."""
    with held_grant_directory() as held:
        yield held


def test_verified_worker_receives_grant_after_registration(grant_dir: tuple[int, Path]) -> None:
    """The launched process is registered first and only then imports nothing but the grant."""
    descriptor, path = grant_dir
    registered: list[str] = []
    with WorkerGrantEndpoint(descriptor, path, NAME) as endpoint:
        child = spawn_worker(endpoint.path, os.getuid())
        expected = expected_for(child)
        identity = endpoint.grant(
            expected,
            deadline=time.monotonic() + 10.0,
            max_refusals=1,
            register=registered.append,
        )
        assert identity == supervisor_identity(child.pid)
        assert registered == [identity]
        assert finish(child) == (0, "granted")
    assert not (path / NAME).exists()


@pytest.mark.parametrize("mismatch", ["pid", "token", "uid"])
def test_unverified_process_is_refused_and_never_granted(
    grant_dir: tuple[int, Path], mismatch: str
) -> None:
    """A connection from any other generation or identity receives EOF, not ready."""
    descriptor, path = grant_dir
    registered: list[str] = []
    with WorkerGrantEndpoint(descriptor, path, NAME) as endpoint:
        child = spawn_worker(endpoint.path, os.getuid())
        actual = expected_for(child)
        expected = ExpectedWorker(
            uid=actual.uid + 1 if mismatch == "uid" else actual.uid,
            pid=os.getpid() if mismatch == "pid" else actual.pid,
            start_token=str(int(actual.start_token) + 1)
            if mismatch == "token"
            else actual.start_token,
        )
        with pytest.raises(TimeoutError, match="did not request its grant"):
            endpoint.grant(
                expected,
                deadline=time.monotonic() + 1.5,
                max_refusals=2,
                register=registered.append,
            )
        assert finish(child) == (3, "RuntimeError")
    assert registered == []
    assert not (path / NAME).exists()


def test_refusal_does_not_consume_the_grant(grant_dir: tuple[int, Path]) -> None:
    """An impostor connecting first is refused; the launched worker is still granted."""
    descriptor, path = grant_dir
    registered: list[str] = []
    with WorkerGrantEndpoint(descriptor, path, NAME) as endpoint:
        impostor = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        impostor.connect(str(endpoint.path))
        child = spawn_worker(endpoint.path, os.getuid())
        try:
            identity = endpoint.grant(
                expected_for(child),
                deadline=time.monotonic() + 10.0,
                max_refusals=1,
                register=registered.append,
            )
            impostor.settimeout(2.0)
            assert impostor.recv(16) == b""
        finally:
            impostor.close()
        assert registered == [identity]
        assert finish(child) == (0, "granted")


def test_registration_failure_withholds_ready(grant_dir: tuple[int, Path]) -> None:
    """A failed durable registration closes the worker connection without a grant."""
    descriptor, path = grant_dir

    def refuse(identity: str) -> None:
        raise ValueError(f"no admitted supervisor for {identity}")

    with WorkerGrantEndpoint(descriptor, path, NAME) as endpoint:
        child = spawn_worker(endpoint.path, os.getuid())
        with pytest.raises(ValueError, match="no admitted supervisor"):
            endpoint.grant(
                expected_for(child),
                deadline=time.monotonic() + 10.0,
                max_refusals=1,
                register=refuse,
            )
        assert finish(child) == (3, "RuntimeError")
    assert not (path / NAME).exists()


def test_worker_refuses_server_with_wrong_identity(grant_dir: tuple[int, Path]) -> None:
    """The worker refuses from listener credentials alone, before any API accept.

    A delivered ``ready`` is therefore not proof that the worker accepted it;
    the supervisor still owns the observed worker outcome.
    """
    descriptor, path = grant_dir
    with WorkerGrantEndpoint(descriptor, path, NAME) as endpoint:
        child = spawn_worker(endpoint.path, os.getuid() + 1)
        assert finish(child) == (3, "PermissionError")


@pytest.mark.parametrize("granted", [True, False])
def test_process_worker_imports_task_only_after_socket_grant(
    grant_dir: tuple[int, Path], granted: bool
) -> None:
    """The actual worker entrypoint gates task import on the socket grant."""
    descriptor, path = grant_dir
    module = task_module(path)
    (path / "work").mkdir()

    def register(identity: str) -> None:
        if not granted:
            raise ValueError("registration refused")

    with WorkerGrantEndpoint(descriptor, path, NAME) as endpoint:
        child = process_worker(
            path,
            ["--grant-socket", str(endpoint.path), "--grant-server-uid", str(os.getuid())],
        )
        try:
            _, pid, token = supervisor_identity(child.pid).split(":", 2)
            expected = ExpectedWorker(uid=os.getuid(), pid=int(pid), start_token=token)
            if granted:
                endpoint.grant(
                    expected, deadline=time.monotonic() + 10.0, max_refusals=1, register=register
                )
            else:
                with pytest.raises(ValueError, match="registration refused"):
                    endpoint.grant(
                        expected,
                        deadline=time.monotonic() + 10.0,
                        max_refusals=1,
                        register=register,
                    )
            _, stderr = child.communicate(timeout=20.0)
        finally:
            assert reap_process_group(child, owned_group_id=child.pid).reaped
    evidence = json.loads((path / "result.json").read_text())
    assert child.returncode == (0 if granted else 1), stderr
    assert evidence["status"] == ("completed" if granted else "failed")
    assert module.with_suffix(".imported").exists() is granted
    if granted:
        assert evidence["result"] == {"granted_execution": 7}
    else:
        assert evidence["error"] == "RuntimeError"


@pytest.mark.parametrize(
    "extra,supervisor",
    [
        (["--grant-socket", "/tmp/" + NAME], True),
        (["--grant-server-uid", "1000"], True),
        (["--grant-socket", "/tmp/" + NAME, "--grant-server-uid", "1000"], False),
    ],
)
def test_process_worker_refuses_incomplete_socket_grant_options(
    grant_dir: tuple[int, Path], extra: list[str], supervisor: bool
) -> None:
    """Partial grant options or a grant without supervisor fail before import."""
    _, path = grant_dir
    module = task_module(path)
    (path / "work").mkdir()
    child = process_worker(path, extra, supervisor=supervisor)
    try:
        _, stderr = child.communicate(timeout=20.0)
    finally:
        assert reap_process_group(child, owned_group_id=child.pid).reaped
    assert child.returncode == 1, stderr
    evidence = json.loads((path / "result.json").read_text())
    assert evidence == {"artifacts": [], "error": "ValueError", "result": {}, "status": "failed"}
    assert not module.with_suffix(".imported").exists()
