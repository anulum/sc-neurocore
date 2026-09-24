# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage peer boundary tests

"""Real Linux peer identity and framed transfer, not OS-isolation qualification."""

from dataclasses import FrozenInstanceError
import os
import socket
import subprocess
import sys
import tempfile
import time

import pytest

from sc_neurocore.studio.platform.storage_peer import (
    read_verified_frame,
    require_storage_peer,
    require_storage_supervisor_identity,
    write_verified_frame,
)
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity


def test_socketpair_reports_creation_identity_without_timeout_change() -> None:
    """Kernel credentials describe this socketpair's actual creating process."""
    left, right = socket.socketpair()
    with left, right:
        left.settimeout(0.3)
        peer = require_storage_peer(left, expected_uid=os.getuid())
        assert (peer.pid, peer.uid, peer.gid) == (os.getpid(), os.getuid(), os.getgid())
        assert left.gettimeout() == 0.3
        with pytest.raises(FrozenInstanceError):
            peer.__setattr__("uid", os.getuid() + 1)


def test_pidfd_binds_live_connected_peer_generation() -> None:
    """A real pidfd and proc start token identify this socket's creator."""
    left, right = socket.socketpair()
    with left, right:
        identity = require_storage_supervisor_identity(left, expected_uid=os.getuid())
        assert identity == supervisor_identity()
        assert left.fileno() >= 0


def test_pidfd_refuses_untrusted_uid_before_generation_read() -> None:
    """A wrong configured UID closes the connection before lease identity use."""
    left, right = socket.socketpair()
    with left, right:
        with pytest.raises(PermissionError):
            require_storage_supervisor_identity(left, expected_uid=os.getuid() + 1)
        assert left.fileno() == -1


@pytest.mark.parametrize("sending", [False, True])
def test_denied_peer_cannot_transfer_frame(sending: bool) -> None:
    """UID denial precedes malformed input parsing and outbound header emission."""
    left, right = socket.socketpair()
    with left, right:
        if not sending:
            right.sendall(b"\xff\xff\xff\xff")
        with pytest.raises(PermissionError):
            if sending:
                write_verified_frame(
                    left,
                    b"secret",
                    expected_uid=os.getuid() + 1,
                    max_bytes=8,
                    deadline=time.monotonic() + 1,
                )
            else:
                read_verified_frame(
                    left, expected_uid=os.getuid() + 1, max_bytes=8, deadline=time.monotonic() + 1
                )
        assert left.fileno() == -1
        if sending:
            assert right.recv(1) == b""


@pytest.mark.parametrize("uid", [-1, True, 0xFFFFFFFF, 2**40])
def test_bad_configuration_preserves_socket(uid: int) -> None:
    """Invalid configured UIDs do not close or consume the connection."""
    left, right = socket.socketpair()
    with left, right:
        right.sendall(b"a")
        with pytest.raises(ValueError, match="expected_uid"):
            require_storage_peer(left, expected_uid=uid)
        assert left.recv(1) == b"a"


@pytest.mark.parametrize(
    "family,kind",
    [
        (socket.AF_INET, socket.SOCK_STREAM),
        (socket.AF_UNIX, socket.SOCK_DGRAM),
        (socket.AF_UNIX, socket.SOCK_STREAM),
    ],
)
def test_unsupported_or_unconnected_socket_refuses(family: int, kind: int) -> None:
    """TCP, datagram and unconnected Unix sockets never supply accepted identity."""
    with socket.socket(family, kind) as channel:
        with pytest.raises(PermissionError):
            require_storage_peer(channel, expected_uid=os.getuid())
        assert channel.fileno() == -1


def test_child_connect_identity_and_bidirectional_verified_frames() -> None:
    """A child connects after launch and both peers verify UID before byte transfer."""
    child_code = """
import socket, sys, time
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
    channel.settimeout(5)
    channel.connect(sys.argv[1])
    deadline = time.monotonic() + 5
    data = read_verified_frame(channel, expected_uid=int(sys.argv[2]), max_bytes=64, deadline=deadline)
    write_verified_frame(channel, data[::-1], expected_uid=int(sys.argv[2]), max_bytes=64, deadline=deadline)
"""
    with (
        tempfile.TemporaryDirectory(prefix="scpeer-") as directory,
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener,
    ):
        endpoint = directory + "/peer.sock"
        listener.bind(endpoint)
        listener.listen(1)
        listener.settimeout(10)
        child = subprocess.Popen(
            [sys.executable, "-c", child_code, endpoint, str(os.getuid())],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            channel, _ = listener.accept()
            with channel:
                peer = require_storage_peer(channel, expected_uid=os.getuid())
                assert peer.pid == child.pid
                assert peer.pid != os.getpid()
                assert require_storage_supervisor_identity(
                    channel, expected_uid=os.getuid()
                ) == supervisor_identity(child.pid)
                deadline = time.monotonic() + 5
                write_verified_frame(
                    channel,
                    b"actual-peer",
                    expected_uid=os.getuid(),
                    max_bytes=64,
                    deadline=deadline,
                )
                assert (
                    read_verified_frame(
                        channel, expected_uid=os.getuid(), max_bytes=64, deadline=deadline
                    )
                    == b"reep-lautca"
                )
            stdout, stderr = child.communicate(timeout=10)
            assert child.returncode == 0, stderr
            assert stdout == ""
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=10)


def test_pidfd_refuses_peer_that_exited_after_connect() -> None:
    """A connection left behind by a dead API process cannot admit a job."""
    child_code = (
        "import socket,sys\nwith socket.socket(socket.AF_UNIX) as s: s.connect(sys.argv[1])\n"
    )
    with (
        tempfile.TemporaryDirectory(prefix="scpeer-exit-") as directory,
        socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener,
    ):
        endpoint = directory + "/peer.sock"
        listener.bind(endpoint)
        listener.listen(1)
        listener.settimeout(5)
        child = subprocess.Popen(
            [sys.executable, "-c", child_code, endpoint],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            channel, _ = listener.accept()
            with channel:
                _, stderr = child.communicate(timeout=5)
                assert child.returncode == 0, stderr
                with pytest.raises(PermissionError):
                    require_storage_supervisor_identity(channel, expected_uid=os.getuid())
                assert channel.fileno() == -1
        finally:
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=5)
