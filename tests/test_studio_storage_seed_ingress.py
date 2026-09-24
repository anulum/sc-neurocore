# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — storage seed ingress tests

"""Exercise bounded seed receipt over real Unix sockets."""

from __future__ import annotations

import os
import socket
import struct
import time
from collections.abc import Mapping
from typing import cast

import pytest

from sc_neurocore.studio.platform.storage_admission_digest import derive_storage_admission_replay
from sc_neurocore.studio.platform.storage_named_tasks import resolve_named_studio_task
from sc_neurocore.studio.platform.storage_seed_ingress import (
    receive_storage_seeds,
    send_storage_seeds,
)
from sc_neurocore.studio.platform.storage_transport import write_frame
from sc_neurocore.studio.platform.policy_models import Principal


def _receive(
    channel: socket.socket,
    manifest: dict[str, int],
    *,
    expected_api_uid: int | None = None,
    frame_max_bytes: int = 4,
    max_seed_bytes: int = 8,
    max_seed_entries: int = 3,
    max_manifest_bytes: int = 32,
    deadline: float | None = None,
) -> dict[str, bytes]:
    return receive_storage_seeds(
        channel,
        manifest=manifest,
        expected_api_uid=os.getuid() if expected_api_uid is None else expected_api_uid,
        frame_max_bytes=frame_max_bytes,
        max_seed_bytes=max_seed_bytes,
        max_seed_entries=max_seed_entries,
        max_manifest_bytes=max_manifest_bytes,
        deadline=time.monotonic() + 1 if deadline is None else deadline,
    )


def test_receives_real_chunks_and_zero_length_seed() -> None:
    """Actual received bytes, not a claimed digest, become replay inputs."""
    reader, writer = socket.socketpair()
    with reader, writer:
        for chunk in (b"ab", b"cde", b"z"):
            write_frame(writer, chunk, max_bytes=4, deadline=time.monotonic() + 1)
        assert _receive(reader, {"b/empty": 0, "a/input": 5, "c/data": 1}) == {
            "a/input": b"abcde",
            "b/empty": b"",
            "c/data": b"z",
        }
        assert reader.fileno() >= 0


def test_sender_and_receiver_transfer_same_seed_bytes() -> None:
    """The API writer and authority reader agree on ordering and chunking."""
    reader, writer = socket.socketpair()
    with reader, writer:
        sent = {"z/empty": b"", "b/data": b"12345", "a/data": b"q"}
        manifest = {"z/empty": 0, "b/data": 5, "a/data": 1}
        send_storage_seeds(
            writer,
            seed_inputs=sent,
            manifest=manifest,
            expected_service_uid=os.getuid(),
            frame_max_bytes=4,
            max_seed_bytes=8,
            max_seed_entries=3,
            max_manifest_bytes=32,
            deadline=time.monotonic() + 1,
        )
        assert _receive(reader, manifest) == sent


def test_sender_refuses_invalid_inputs_before_transfer() -> None:
    """A bad path or byte type does not emit even a frame header."""
    reader, writer = socket.socketpair()
    with reader, writer:
        for seeds in ({"../bad": b"x"}, {"good": cast(bytes, bytearray(b"x"))}):
            with pytest.raises(ValueError):
                send_storage_seeds(
                    writer,
                    seed_inputs=seeds,
                    manifest={name: len(data) for name, data in seeds.items()},
                    expected_service_uid=os.getuid(),
                    frame_max_bytes=4,
                    max_seed_bytes=8,
                    max_seed_entries=3,
                    max_manifest_bytes=32,
                    deadline=time.monotonic() + 1,
                )
        reader.settimeout(0.02)
        with pytest.raises(TimeoutError):
            reader.recv(1)


def test_empty_sender_still_verifies_service_uid() -> None:
    """A seed-free admission cannot skip authority peer verification."""
    reader, writer = socket.socketpair()
    with reader, writer, pytest.raises(PermissionError):
        send_storage_seeds(
            writer,
            seed_inputs={},
            manifest={},
            expected_service_uid=os.getuid() + 1,
            frame_max_bytes=4,
            max_seed_bytes=8,
            max_seed_entries=3,
            max_manifest_bytes=32,
            deadline=time.monotonic() + 1,
        )


def test_sender_rejects_invalid_service_uid_or_mapping() -> None:
    """Bad local preconditions do not send bytes or consume a socket."""
    reader, writer = socket.socketpair()
    with reader, writer:
        with pytest.raises(ValueError, match="named bytes"):
            send_storage_seeds(
                writer,
                seed_inputs=cast(Mapping[str, bytes], []),
                manifest={},
                expected_service_uid=os.getuid(),
                frame_max_bytes=4,
                max_seed_bytes=8,
                max_seed_entries=3,
                max_manifest_bytes=32,
                deadline=time.monotonic() + 1,
            )
        with pytest.raises(ValueError, match="service UID"):
            send_storage_seeds(
                writer,
                seed_inputs={},
                manifest={},
                expected_service_uid=-1,
                frame_max_bytes=4,
                max_seed_bytes=8,
                max_seed_entries=3,
                max_manifest_bytes=32,
                deadline=time.monotonic() + 1,
            )
        with pytest.raises(ValueError, match="manifest"):
            send_storage_seeds(
                writer,
                seed_inputs={},
                manifest=cast(Mapping[str, int], []),
                expected_service_uid=os.getuid(),
                frame_max_bytes=4,
                max_seed_bytes=8,
                max_seed_entries=3,
                max_manifest_bytes=32,
                deadline=time.monotonic() + 1,
            )
        assert writer.fileno() >= 0


@pytest.mark.parametrize("manifest", [{"a": 2}, {"b": 1}, {}])
def test_sender_rejects_metadata_content_mismatch(manifest: dict[str, int]) -> None:
    """Names and sizes sent in request metadata must describe actual bytes."""
    reader, writer = socket.socketpair()
    with reader, writer:
        with pytest.raises(ValueError, match="manifest"):
            send_storage_seeds(
                writer,
                seed_inputs={"a": b"x"},
                manifest=manifest,
                expected_service_uid=os.getuid(),
                frame_max_bytes=4,
                max_seed_bytes=8,
                max_seed_entries=3,
                max_manifest_bytes=32,
                deadline=time.monotonic() + 1,
            )
        reader.settimeout(0.02)
        with pytest.raises(TimeoutError):
            reader.recv(1)


@pytest.mark.parametrize(
    "change",
    [
        {"frame_max_bytes": 0},
        {"max_seed_bytes": -1},
        {"max_seed_entries": -1},
        {"max_manifest_bytes": 0},
        {"max_seed_entries": 0},
        {"max_manifest_bytes": 1},
        {"max_seed_bytes": 0},
    ],
)
def test_receiver_rejects_invalid_or_exceeded_limits(change: dict[str, int]) -> None:
    """Invalid budgets and declared content refuse before a wire read."""
    reader, writer = socket.socketpair()
    with reader, writer:
        limits = {
            "frame_max_bytes": 4,
            "max_seed_bytes": 8,
            "max_seed_entries": 3,
            "max_manifest_bytes": 32,
        }
        limits.update(change)
        with pytest.raises(ValueError):
            _receive(reader, {"ab": 1}, **limits)
        assert reader.fileno() >= 0


def test_receiver_rejects_bad_uid_and_manifest_type() -> None:
    """Invalid caller configuration never probes an ambiguous stream."""
    reader, writer = socket.socketpair()
    with reader, writer:
        with pytest.raises(ValueError, match="API UID"):
            _receive(reader, {}, expected_api_uid=-1)
        with pytest.raises(ValueError, match="manifest"):
            _receive(reader, cast(dict[str, int], []))
        assert reader.fileno() >= 0


def test_received_seed_bytes_change_durable_replay_identity() -> None:
    """The digest follows wire bytes even when metadata and mutation ID match."""
    digests: list[str] = []
    for actual in (b"first", b"other"):
        reader, writer = socket.socketpair()
        with reader, writer:
            write_frame(writer, actual, max_bytes=8, deadline=time.monotonic() + 1)
            received = _receive(reader, {"data.bin": 5}, frame_max_bytes=8)
        replay = derive_storage_admission_replay(
            requester=Principal("operator", frozenset({"studio.admin"})),
            mutation_id="one-mutation",
            workspace="default",
            task=resolve_named_studio_task("analysis.run", authorized_route="/api/analysis/jobs"),
            authorized_route="/api/analysis/jobs",
            payload_json=b'{"model":"lif"}',
            seed_inputs=received,
            execution_timeout_seconds=10.0,
            queue_wait_seconds=None,
            admission=None,
            training_config=None,
            experiment_sha256=None,
            max_metadata_bytes=4096,
            max_seed_bytes=8,
            max_seed_entries=1,
        )
        digests.append(replay.payload_sha256)
    assert digests[0] != digests[1]


@pytest.mark.parametrize(
    "manifest",
    [
        {"../outside": 1},
        {"/absolute": 1},
        {"a//b": 1},
        {"a\ud800": 1},
        {"bad\nname": 1},
        {"a": -1},
        {"a": True},
        {"a": 9},
        {"a": 5, "b": 4},
        {"a": 0, "b": 0, "c": 0, "d": 0},
        {"long-name-exceeds-32-byte-manifest-limit": 0},
    ],
)
def test_invalid_manifest_refuses_before_any_receive(manifest: dict[str, int]) -> None:
    """Malformed or oversized declarations cannot initiate a payload read."""
    reader, writer = socket.socketpair()
    with reader, writer:
        with pytest.raises(ValueError):
            _receive(reader, manifest)
        assert reader.fileno() >= 0


def test_oversized_chunk_closes_ambiguous_connection() -> None:
    """A frame cannot exceed the declared remaining bytes."""
    reader, writer = socket.socketpair()
    with reader, writer:
        writer.sendall(struct.pack("!I", 3) + b"abc")
        with pytest.raises(ValueError, match="declared"):
            _receive(reader, {"a": 2})
        assert reader.fileno() == -1


def test_zero_byte_manifest_still_verifies_peer_uid() -> None:
    """No payload frame does not bypass connection identity."""
    reader, writer = socket.socketpair()
    with reader, writer, pytest.raises(PermissionError):
        _receive(reader, {"empty": 0}, expected_api_uid=os.getuid() + 1)


@pytest.mark.parametrize("deadline", [True, float("nan"), float("inf"), 10**1000])
def test_zero_byte_manifest_rejects_invalid_deadline(deadline: float) -> None:
    """An empty transfer cannot bypass the absolute deadline contract."""
    reader, writer = socket.socketpair()
    with reader, writer, pytest.raises(ValueError, match="deadline"):
        _receive(reader, {"empty": 0}, deadline=deadline)


def test_zero_byte_manifest_rejects_expired_deadline() -> None:
    """Already expired work refuses even if it would receive no frames."""
    reader, writer = socket.socketpair()
    with reader, writer, pytest.raises(TimeoutError, match="deadline"):
        _receive(reader, {"empty": 0}, deadline=time.monotonic() - 1)


def test_truncated_seed_closes_ambiguous_connection() -> None:
    """A valid prefix cannot be mistaken for a complete seed."""
    reader, writer = socket.socketpair()
    with reader, writer:
        write_frame(writer, b"a", max_bytes=4, deadline=time.monotonic() + 1)
        writer.shutdown(socket.SHUT_WR)
        with pytest.raises(EOFError):
            _receive(reader, {"a": 2})
        assert reader.fileno() == -1


def test_silent_peer_expires_shared_deadline() -> None:
    """The receive loop never renews a silent peer's budget."""
    reader, writer = socket.socketpair()
    with reader, writer:
        with pytest.raises(TimeoutError):
            _receive(reader, {"a": 1}, deadline=time.monotonic() + 0.02)
        assert reader.fileno() == -1
