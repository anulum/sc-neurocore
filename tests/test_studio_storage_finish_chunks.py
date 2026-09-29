# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chunked finish byte custody

"""Transfer actual frame chunks and refuse truncation, wrong boundaries and digests."""

import hashlib
import os
import socket
import time

import pytest

from sc_neurocore.studio.platform.storage_finish_chunks import (
    FinishChunkMismatch,
    receive_finish_artifact,
)
from sc_neurocore.studio.platform.storage_peer import write_verified_frame
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from tests.studio_storage_finish_support import finish, request, started, stop
from tests.studio_storage_supervision_support import ledger as ledger
from tests.studio_storage_supervision_support import clock as clock


@pytest.mark.parametrize("size", [0, 1, 64, 129])
def test_exact_chunk_boundaries_preserve_all_bytes(size: int) -> None:
    """Empty, exact and partial final frames have one unambiguous byte sequence."""
    payload = bytes(range(size))
    deadline = time.monotonic() + 5
    client, service = socket.socketpair()
    with client, service:
        for offset in range(0, size, 64):
            write_verified_frame(
                client,
                payload[offset : offset + 64],
                expected_uid=os.getuid(),
                max_bytes=64,
                deadline=deadline,
            )
        assert (
            receive_finish_artifact(
                service,
                size_bytes=size,
                sha256=hashlib.sha256(payload).hexdigest(),
                expected_api_uid=os.getuid(),
                frame_max_bytes=64,
                deadline=deadline,
            )
            == payload
        )


@pytest.mark.parametrize("damage", ["short", "digest", "closed"])
def test_invalid_chunk_content_is_never_returned(damage: str) -> None:
    """A premature remainder, altered bytes or disconnected peer cannot satisfy custody."""
    client, service = socket.socketpair()
    deadline = time.monotonic() + 5
    with client, service:
        if damage != "closed":
            write_verified_frame(
                client,
                b"changed" if damage == "digest" else b"x",
                expected_uid=os.getuid(),
                max_bytes=64,
                deadline=deadline,
            )
        else:
            client.shutdown(socket.SHUT_WR)
        with pytest.raises(EOFError if damage == "closed" else FinishChunkMismatch):
            receive_finish_artifact(
                service,
                size_bytes=7,
                sha256=hashlib.sha256(b"correct").hexdigest(),
                expected_api_uid=os.getuid(),
                frame_max_bytes=64,
                deadline=deadline,
            )


@pytest.mark.parametrize("size,frame", [(-1, 64), (True, 64), (0, True), (0, 0), (0, 1 << 32)])
def test_invalid_transfer_limits_refuse(size: int, frame: int) -> None:
    """Impossible sizes and frame ceilings fail before any socket read."""
    client, service = socket.socketpair()
    with client, service, pytest.raises(ValueError):
        receive_finish_artifact(
            service,
            size_bytes=size,
            sha256=hashlib.sha256(b"").hexdigest(),
            expected_api_uid=os.getuid(),
            frame_max_bytes=frame,
            deadline=time.monotonic() + 1,
        )


def test_large_finish_is_sealed_and_replays_without_retransmission(ledger: StudioJobLedger) -> None:
    """Actual authority stores multi-frame content and an identical replay seals nothing new."""
    stop(started(ledger))
    payload = bytes(range(256)) * 100
    sent = request({"checkpoint.bin": payload})
    assert finish(ledger, sent, [payload], frame_max_bytes=4096).reply == "sealed"
    assert (ledger.path.parent / sent.job_id / "checkpoint.bin").read_bytes() == payload
    assert finish(ledger, sent, [payload], frame_max_bytes=4096).reply == "already_sealed"
