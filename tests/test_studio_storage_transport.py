# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage framing socket tests

"""Exercise storage byte framing with real Unix stream sockets."""

import socket
import struct
import threading
import time

import pytest

from sc_neurocore.studio.platform.storage_transport import read_frame, write_frame


def test_multiple_bidirectional_frames_restore_timeout() -> None:
    """Both directions preserve frame boundaries and caller timeouts."""
    left, right = socket.socketpair()
    with left, right:
        left.settimeout(0.7)
        right.settimeout(0.9)
        for sender, receiver in ((left, right), (right, left)):
            for payload in (b"a", b"second", b"\x00\xff"):
                deadline = time.monotonic() + 2
                write_frame(sender, payload, max_bytes=6, deadline=deadline)
                assert read_frame(receiver, max_bytes=6, deadline=deadline) == payload
        assert left.gettimeout() == 0.7
        assert right.gettimeout() == 0.9


@pytest.mark.parametrize("size", [0, 1025, 0xFFFFFFFF])
def test_invalid_header_refuses_without_waiting_for_body(size: int) -> None:
    """A header alone suffices to reject an invalid allocation request."""
    reader, writer = socket.socketpair()
    with reader, writer:
        writer.sendall(struct.pack("!I", size))
        with pytest.raises(ValueError, match="declared"):
            read_frame(reader, max_bytes=1024, deadline=time.monotonic() + 1)
        assert reader.fileno() == -1


@pytest.mark.parametrize("partial", [b"", b"\x00\x00", struct.pack("!I", 5) + b"ab"])
def test_truncated_frame_closes_connection(partial: bytes) -> None:
    """EOF cannot leave an ambiguous connection available for reuse."""
    reader, writer = socket.socketpair()
    with reader, writer:
        writer.sendall(partial)
        writer.shutdown(socket.SHUT_WR)
        with pytest.raises(EOFError):
            read_frame(reader, max_bytes=8, deadline=time.monotonic() + 1)
        assert reader.fileno() == -1


@pytest.mark.parametrize("cap", [0, -1, True, 2**32])
def test_invalid_cap_preserves_socket(cap: int) -> None:
    """Argument errors do not mutate the stream or its configured timeout."""
    reader, writer = socket.socketpair()
    with reader, writer:
        reader.settimeout(0.3)
        with pytest.raises(ValueError, match="max_bytes"):
            read_frame(reader, max_bytes=cap, deadline=time.monotonic() + 1)
        assert reader.fileno() >= 0
        assert reader.gettimeout() == 0.3


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), True, 10**1000, 1e300])
def test_invalid_deadline_preserves_socket(deadline: float) -> None:
    """Nonfinite times and booleans are not valid deadline contracts."""
    reader, writer = socket.socketpair()
    with reader, writer:
        with pytest.raises(ValueError, match="deadline"):
            read_frame(reader, max_bytes=8, deadline=deadline)
        assert reader.fileno() >= 0


def test_silent_peer_deadline() -> None:
    """An idle open peer cannot retain the reader indefinitely."""
    reader, writer = socket.socketpair()
    with reader, writer:
        with pytest.raises(TimeoutError):
            read_frame(reader, max_bytes=8, deadline=time.monotonic() + 0.05)
        assert reader.fileno() == -1


def test_backpressured_sender_deadline() -> None:
    """A nonreading peer cannot retain a large send beyond its deadline."""
    reader, writer = socket.socketpair()
    with reader, writer:
        writer.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        with pytest.raises(TimeoutError):
            write_frame(
                writer,
                b"x" * (1024 * 1024),
                max_bytes=1024 * 1024,
                deadline=time.monotonic() + 0.05,
            )
        assert writer.fileno() == -1


@pytest.mark.parametrize("trickle", [False, True])
def test_fragmentation_does_not_renew_deadline(trickle: bool) -> None:
    """Fragmented success and slow-peer expiry share one absolute budget."""
    reader, writer = socket.socketpair()
    stop = threading.Event()

    def produce() -> None:
        try:
            for byte in struct.pack("!I", 8) + b"abcdefgh":
                writer.sendall(bytes([byte]))
                if stop.wait(0.03 if trickle else 0.001):
                    return
        except (BrokenPipeError, ConnectionResetError):
            return

    with reader, writer:
        worker = threading.Thread(target=produce)
        worker.start()
        try:
            deadline = time.monotonic() + (0.1 if trickle else 2)
            if trickle:
                with pytest.raises(TimeoutError):
                    read_frame(reader, max_bytes=8, deadline=deadline)
                assert reader.fileno() == -1
            else:
                assert read_frame(reader, max_bytes=8, deadline=deadline) == b"abcdefgh"
        finally:
            stop.set()
            worker.join(timeout=2)
            assert not worker.is_alive()


@pytest.mark.parametrize(
    "family,kind", [(socket.AF_INET, socket.SOCK_STREAM), (socket.AF_UNIX, socket.SOCK_DGRAM)]
)
def test_wrong_socket_contract_preserves_socket(family: int, kind: int) -> None:
    """Unsupported transports refuse without binding, connecting or closing."""
    with socket.socket(family, kind) as channel:
        channel.settimeout(0.4)
        with pytest.raises(ValueError, match="requires"):
            read_frame(channel, max_bytes=8, deadline=time.monotonic() + 1)
        assert channel.fileno() >= 0
        assert channel.gettimeout() == 0.4


@pytest.mark.parametrize("payload", [b"", b"oversized"])
def test_invalid_payload_leaves_no_wire_bytes(payload: bytes) -> None:
    """Rejected sends leave the socket reusable and do not send a header."""
    reader, writer = socket.socketpair()
    with reader, writer:
        with pytest.raises(ValueError, match="payload"):
            write_frame(writer, payload, max_bytes=2, deadline=time.monotonic() + 1)
        reader.setblocking(False)
        with pytest.raises(BlockingIOError):
            reader.recv(1)
        deadline = time.monotonic() + 1
        write_frame(writer, b"ok", max_bytes=2, deadline=deadline)
        assert read_frame(reader, max_bytes=2, deadline=deadline) == b"ok"
        assert reader.gettimeout() == 0.0


@pytest.mark.parametrize("sending", [False, True])
def test_expired_deadline_closes_without_transferring(sending: bool) -> None:
    """A deadline already in the past fails before the first wire operation."""
    reader, writer = socket.socketpair()
    with reader, writer:
        with pytest.raises(TimeoutError, match="expired"):
            if sending:
                write_frame(writer, b"a", max_bytes=1, deadline=time.monotonic() - 1)
            else:
                read_frame(writer, max_bytes=1, deadline=time.monotonic() - 1)
        assert writer.fileno() == -1
        assert reader.recv(1) == b""


def test_closed_peer_send_failure_closes_writer() -> None:
    """A disconnected peer produces an error, not a fabricated send receipt."""
    reader, writer = socket.socketpair()
    reader.close()
    with writer:
        with pytest.raises(BrokenPipeError):
            write_frame(writer, b"a", max_bytes=1, deadline=time.monotonic() + 1)
        assert writer.fileno() == -1
