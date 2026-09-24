# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — launcher-started worker registration grant

"""The grant endpoint verifies, bounds and cleans up without adopting foreign state."""

from __future__ import annotations

import errno
import os
import shutil
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.storage_worker_grant import (
    ExpectedWorker,
    WorkerGrantEndpoint,
    receive_socket_grant,
    validate_grant_name,
)
from tests.studio_seccomp_support import SECCOMP_AVAILABLE, Refusal, run_refused
from tests.studio_storage_grant_support import *


@pytest.fixture
def grant_dir() -> Iterator[tuple[int, Path]]:
    """Hold a short private directory so the socket path fits the Unix limit."""
    with held_grant_directory() as held:
        yield held


def test_refusal_budget_is_bounded(grant_dir: tuple[int, Path]) -> None:
    """Repeated unverified connections cannot hold the endpoint open indefinitely."""
    descriptor, path = grant_dir
    impostors = [socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) for _ in range(2)]
    try:
        with WorkerGrantEndpoint(descriptor, path, NAME) as endpoint:
            for impostor in impostors:
                impostor.connect(str(endpoint.path))
            expected = ExpectedWorker(uid=os.getuid(), pid=1, start_token="1")
            with pytest.raises(PermissionError, match="refused unverified connections"):
                endpoint.grant(
                    expected,
                    deadline=time.monotonic() + 5.0,
                    max_refusals=1,
                    register=lambda identity: None,
                )
    finally:
        for impostor in impostors:
            impostor.close()
    assert not (path / NAME).exists()


def test_deadline_without_connection_closes_endpoint(grant_dir: tuple[int, Path]) -> None:
    """No worker within the deadline is a timeout and the endpoint is removed."""
    descriptor, path = grant_dir
    endpoint = WorkerGrantEndpoint(descriptor, path, NAME)
    endpoint.open()
    assert (path / NAME).is_socket()
    with pytest.raises(TimeoutError):
        endpoint.grant(
            ExpectedWorker(uid=os.getuid(), pid=1, start_token="1"),
            deadline=time.monotonic() + 0.2,
            max_refusals=1,
            register=lambda identity: None,
        )
    assert not (path / NAME).exists()
    with pytest.raises(RuntimeError, match="not open"):
        endpoint.grant(
            ExpectedWorker(uid=os.getuid(), pid=1, start_token="1"),
            deadline=time.monotonic() + 0.2,
            max_refusals=1,
            register=lambda identity: None,
        )


def test_existing_entry_is_never_adopted(grant_dir: tuple[int, Path]) -> None:
    """A stale or planted entry with the endpoint name refuses and stays untouched."""
    descriptor, path = grant_dir
    planted = path / NAME
    planted.write_text("planted")
    with pytest.raises(FileExistsError):
        WorkerGrantEndpoint(descriptor, path, NAME).open()
    assert planted.read_text() == "planted"


def test_substituted_directory_refuses(grant_dir: tuple[int, Path]) -> None:
    """The held directory must still be the canonical path when binding."""
    descriptor, path = grant_dir
    moved = path.with_name(path.name + "m")
    path.rename(moved)
    path.mkdir()
    try:
        with pytest.raises(PermissionError, match="directory changed"):
            WorkerGrantEndpoint(descriptor, path, NAME).open()
        assert list(path.iterdir()) == []
        assert list(moved.iterdir()) == []
    finally:
        shutil.rmtree(moved)


def test_close_removes_only_its_own_inode(grant_dir: tuple[int, Path]) -> None:
    """A replacement entry created after bind survives endpoint shutdown."""
    descriptor, path = grant_dir
    endpoint = WorkerGrantEndpoint(descriptor, path, NAME)
    endpoint.open()
    (path / NAME).unlink()
    (path / NAME).write_text("replacement")
    endpoint.close()
    assert (path / NAME).read_text() == "replacement"
    endpoint.close()


def test_endpoint_lifecycle_and_argument_contracts(grant_dir: tuple[int, Path]) -> None:
    """Double open, invalid budgets, long paths and malformed values refuse."""
    descriptor, path = grant_dir
    with WorkerGrantEndpoint(descriptor, path, NAME) as endpoint:
        with pytest.raises(RuntimeError, match="already open"):
            endpoint.open()
        invalid_budgets: tuple[object, ...] = (0, -1, True, 1.0)
        for budget in invalid_budgets:
            with pytest.raises(ValueError, match="refusal budget"):
                endpoint.grant(
                    ExpectedWorker(uid=os.getuid(), pid=1, start_token="1"),
                    deadline=time.monotonic() + 0.1,
                    max_refusals=cast(int, budget),
                    register=lambda identity: None,
                )
    deep = Path("/" + "d" * 100)
    with pytest.raises(ValueError, match="Unix limit"):
        WorkerGrantEndpoint(descriptor, deep, NAME).open()
    for name in ("grant-" + "a" * 32 + ".sock", "Grant.sock", "../" + NAME, NAME + "x"):
        with pytest.raises(ValueError, match="name is invalid"):
            validate_grant_name(name)
    for uid, pid, token in ((0, 1, "1"), (1, 0, "1"), (1, 1, "0"), (1, 1, "x1"), (1, 1, "01")):
        with pytest.raises(ValueError):
            ExpectedWorker(uid=uid, pid=pid, start_token=token)


def test_worker_side_path_contract(grant_dir: tuple[int, Path]) -> None:
    """Relative, misnamed and missing endpoints refuse before any grant."""
    _, path = grant_dir
    with pytest.raises(ValueError, match="absolute"):
        receive_socket_grant(Path(NAME), expected_server_uid=os.getuid())
    with pytest.raises(ValueError, match="name is invalid"):
        receive_socket_grant(path / "worker.sock", expected_server_uid=os.getuid())
    with pytest.raises(FileNotFoundError):
        receive_socket_grant(path / NAME, expected_server_uid=os.getuid())


def test_in_process_worker_side_accepts_exact_grant(grant_dir: tuple[int, Path]) -> None:
    """The worker-side reader accepts the exact grant from the configured server identity."""
    descriptor, path = grant_dir
    failures: list[BaseException] = []
    with WorkerGrantEndpoint(descriptor, path, NAME) as endpoint:

        def connect() -> None:
            try:
                receive_socket_grant(endpoint.path, expected_server_uid=os.getuid())
            except BaseException as exc:
                failures.append(exc)

        worker = threading.Thread(target=connect)
        worker.start()
        _, pid, token = supervisor_identity().split(":", 2)
        identity = endpoint.grant(
            ExpectedWorker(uid=os.getuid(), pid=int(pid), start_token=token),
            deadline=time.monotonic() + 10.0,
            max_refusals=1,
            register=lambda observed: None,
        )
        worker.join(timeout=10.0)
    assert not worker.is_alive()
    assert failures == []
    assert identity == supervisor_identity()


def test_registration_past_deadline_withholds_ready(grant_dir: tuple[int, Path]) -> None:
    """A registration that outlives the grant deadline never releases the worker."""
    descriptor, path = grant_dir
    failures: list[BaseException] = []
    registered: list[str] = []

    def slow_register(identity: str) -> None:
        time.sleep(0.6)
        registered.append(identity)

    with WorkerGrantEndpoint(descriptor, path, NAME) as endpoint:

        def connect() -> None:
            try:
                receive_socket_grant(endpoint.path, expected_server_uid=os.getuid())
            except BaseException as exc:
                failures.append(exc)

        worker = threading.Thread(target=connect)
        worker.start()
        _, pid, token = supervisor_identity().split(":", 2)
        with pytest.raises(TimeoutError, match="expired before ready"):
            endpoint.grant(
                ExpectedWorker(uid=os.getuid(), pid=int(pid), start_token=token),
                deadline=time.monotonic() + 0.3,
                max_refusals=1,
                register=slow_register,
            )
        worker.join(timeout=10.0)
    assert len(registered) == 1
    assert len(failures) == 1 and isinstance(failures[0], RuntimeError)
    assert not (path / NAME).exists()


@pytest.mark.skipif(not SECCOMP_AVAILABLE, reason="seccomp filters need Linux x86_64")
def test_refused_mode_change_after_bind_removes_the_new_endpoint() -> None:
    """A kernel-refused mode change closes and unlinks the endpoint it bound."""
    result = run_refused(
        "import errno, os\n"
        "from tests.studio_storage_grant_support import NAME, held_grant_directory\n"
        "from sc_neurocore.studio.platform.storage_worker_grant import WorkerGrantEndpoint\n"
        "with held_grant_directory() as (descriptor, path):\n"
        "    endpoint = WorkerGrantEndpoint(descriptor, path, NAME)\n"
        "    install_refusals(REFUSALS)\n"
        "    try:\n"
        "        endpoint.open()\n"
        "        refused = None\n"
        "    except PermissionError as error:\n"
        "        refused = error.errno\n"
        "    left = sorted(os.listdir(path))\n"
        "    endpoint.close()\n"
        "print(json.dumps({'errno': refused, 'left': left}))\n",
        [Refusal("fchmodat", errno.EPERM), Refusal("fchmodat2", errno.EPERM)],
    )
    assert result == {"errno": errno.EPERM, "left": []}


def test_refused_bind_leaves_nothing_behind(grant_dir: tuple[int, Path]) -> None:
    """A directory the API cannot write refuses the bind before any entry exists."""
    descriptor, path = grant_dir
    path.chmod(0o500)
    try:
        endpoint = WorkerGrantEndpoint(descriptor, path, NAME)
        with pytest.raises(PermissionError):
            endpoint.open()
        assert os.listdir(path) == []
        endpoint.close()
    finally:
        path.chmod(0o700)


def test_close_tolerates_an_already_removed_endpoint(grant_dir: tuple[int, Path]) -> None:
    """An endpoint removed by its directory owner closes without error."""
    descriptor, path = grant_dir
    endpoint = WorkerGrantEndpoint(descriptor, path, NAME)
    endpoint.open()
    (path / NAME).unlink()
    endpoint.close()
    assert not (path / NAME).exists()
