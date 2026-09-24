# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — delegated job finish over broken transfers and crashes

"""Broken transfers, foreign peers and a crashed authority seal nothing wrong.

Frames travel over real socket pairs; a foreign identity is refused before
any frame; a transfer cut after ``ready`` leaves the job live; the storage
process killed by the kernel between sealing and commit is completed by an
identical retry.
"""

from __future__ import annotations

import contextlib
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.storage_finish import serve_finish
from sc_neurocore.studio.platform.storage_finish_client import exchange_finish
from sc_neurocore.studio.platform.storage_finish_protocol import (
    decode_finish_response,
    encode_finish_message,
)
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame
from tests.studio_seccomp_support import REPOSITORY, SECCOMP_AVAILABLE
from tests.studio_storage_finish_support import FILES, finish, request, reserved, started, stop
from tests.studio_storage_supervision_support import *


def _serve(ledger: StudioJobLedger, channel: socket.socket, *, api_uid: int) -> None:
    serve_finish(
        channel,
        ledger=ledger,
        workspace="default",
        expected_api_uid=api_uid,
        frame_max_bytes=4096,
        max_artifact_bytes=65536,
        max_artifact_entries=16,
        deadline=time.monotonic() + 10,
    )


def test_a_foreign_peer_identity_is_refused_on_both_sides(ledger: StudioJobLedger) -> None:
    """Neither side sends a frame to a peer that is not the configured identity."""
    stop(started(ledger))
    service, client = socket.socketpair()
    with client, pytest.raises(PermissionError):
        _serve(ledger, service, api_uid=os.getuid() + 1)
    service, client = socket.socketpair()
    with service:
        with pytest.raises(PermissionError):
            exchange_finish(
                client,
                request(FILES),
                list(FILES.values()),
                expected_service_uid=os.getuid() + 1,
                max_bytes=4096,
                deadline=time.monotonic() + 5,
            )
        service.settimeout(5.0)
        assert service.recv(16) == b""
    assert (ledger.record(JOB).status, reserved(ledger)) == ("running", 1)
    assert not (ledger.path.parent / JOB).exists()


@pytest.mark.parametrize(
    "damage,error",
    [("closed", EOFError), ("truncated", EOFError), ("oversize", ValueError)],
)
def test_a_transfer_broken_after_ready_seals_nothing(
    ledger: StudioJobLedger, damage: str, error: type[Exception]
) -> None:
    """A peer that stops, cuts a frame short or overruns the ceiling leaves the job live."""
    stop(started(ledger))
    service, client = socket.socketpair()
    failures: list[BaseException] = []

    def serve() -> None:
        try:
            _serve(ledger, service, api_uid=os.getuid())
        except BaseException as exc:
            failures.append(exc)

    thread = threading.Thread(target=serve)
    thread.start()
    sent = request(FILES)
    deadline = time.monotonic() + 10

    def send(payload: bytes, ceiling: int = 4096) -> None:
        write_verified_frame(
            client, payload, expected_uid=os.getuid(), max_bytes=ceiling, deadline=deadline
        )

    with client:
        send(encode_finish_message(sent))
        ready = read_verified_frame(
            client, expected_uid=os.getuid(), max_bytes=4096, deadline=deadline
        )
        assert decode_finish_response(ready, request=sent, max_bytes=4096).reply == "ready"
        if damage == "truncated":
            client.sendall(struct.pack("!I", 12) + b"partial")
        elif damage == "oversize":
            # The authority refuses the header and closes while the body is
            # still being written, so the writer may see the closed peer.
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                send(b"x" * 5000, 8192)
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert [type(failure) for failure in failures] == [error]
    assert (ledger.record(JOB).status, reserved(ledger)) == ("running", 1)
    assert not (ledger.path.parent / JOB).exists()


@pytest.mark.skipif(not SECCOMP_AVAILABLE, reason="held system calls need Linux x86_64")
def test_authority_crash_between_seal_and_commit_is_completed_by_a_retry(
    ledger: StudioJobLedger,
) -> None:
    """A storage service killed after sealing, before its commit, loses nothing.

    The service runs as its own process on the API's socket. The kernel holds
    its first write to the ledger's WAL, the commit of the terminal record and
    the reservation release, and the process is killed there. The API sees an
    ambiguous failure and retries.
    """
    stop(started(ledger))
    service, client = socket.socketpair()
    program = (
        "import os, signal, socket, sys, time\n"
        "from pathlib import Path\n"
        "from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger\n"
        "from sc_neurocore.studio.platform.storage_finish import serve_finish\n"
        "from tests.studio_syscall_support import hold_system_calls\n"
        "channel = socket.socket(fileno=int(sys.argv[2]))\n"
        "ledger = StudioJobLedger(root=Path(sys.argv[1]), supervisor='storage:2:2')\n"
        "def decide(call):\n"
        "    if call.name == 'pwrite64' and call.descriptor_path(0).endswith('-wal'):\n"
        "        os.kill(os.getpid(), signal.SIGKILL)\n"
        "hold_system_calls(['pwrite64'], decide)\n"
        "serve_finish(channel, ledger=ledger, workspace='default',\n"
        "    expected_api_uid=os.getuid(), frame_max_bytes=4096, max_artifact_bytes=65536,\n"
        "    max_artifact_entries=16, deadline=time.monotonic() + 30)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", program, str(ledger.path.parent), str(service.fileno())],
        pass_fds=(service.fileno(),),
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(REPOSITORY / "src"), str(REPOSITORY))),
        },
        cwd=REPOSITORY,
    )
    service.close()
    with pytest.raises((EOFError, ConnectionError)):
        exchange_finish(
            client,
            request(FILES),
            list(FILES.values()),
            expected_service_uid=os.getuid(),
            max_bytes=4096,
            deadline=time.monotonic() + 30,
        )
    assert child.wait(timeout=30) == -signal.SIGKILL
    assert (ledger.record(JOB).status, reserved(ledger)) == ("running", 1)
    assert (ledger.path.parent / JOB / "weights.bin").read_bytes() == FILES["weights.bin"]
    retry = finish(ledger, request(FILES, request_id="e" * 32), list(FILES.values()))
    assert (retry.reply, retry.reason) == ("sealed", None)
    assert ledger.record(JOB).status == "completed"
    assert reserved(ledger) == 0
