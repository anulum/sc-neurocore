# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — file-backed seed ingress tests

"""Prove real Unix transfer into private files and failure cleanup."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import socket
import time

import pytest

from sc_neurocore.studio.platform.jobs_admission_replay import StorageAdmissionReplay
from sc_neurocore.studio.platform.policy_models import Principal
from sc_neurocore.studio.platform.storage_admission_digest import derive_storage_admission_replay
from sc_neurocore.studio.platform.storage_named_tasks import resolve_named_studio_task
from sc_neurocore.studio.platform.storage_seed_files import (
    ReceivedStorageSeedFile,
    receive_storage_seed_files,
)
from sc_neurocore.studio.platform.storage_transport import write_frame


def test_received_seed_files_are_unlinked_and_close_after_use(tmp_path: Path) -> None:
    """Actual wire bytes determine digests without retaining a seed dictionary."""
    rootfd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    reader, writer = socket.socketpair()
    try:
        with reader, writer:
            deadline = time.monotonic() + 2
            for chunk in (b"abc", b"de", b"z"):
                write_frame(writer, chunk, max_bytes=4, deadline=deadline)
            with receive_storage_seed_files(
                reader,
                manifest={"b/empty": 0, "a/input": 5, "c/data": 1},
                expected_api_uid=os.getuid(),
                authority_dirfd=rootfd,
                frame_max_bytes=4,
                max_seed_bytes=8,
                max_seed_entries=3,
                max_manifest_bytes=32,
                deadline=deadline,
            ) as received:
                assert [(seed.name, seed.size, seed.sha256) for seed in received.files] == [
                    ("a/input", 5, hashlib.sha256(b"abcde").hexdigest()),
                    ("b/empty", 0, hashlib.sha256(b"").hexdigest()),
                    ("c/data", 1, hashlib.sha256(b"z").hexdigest()),
                ]
                assert [seed.stream.read() for seed in received.files] == [b"abcde", b"", b"z"]
                assert all(os.fstat(seed.stream.fileno()).st_nlink == 0 for seed in received.files)
                streams = [seed.stream for seed in received.files]
            assert all(stream.closed for stream in streams)
            assert list(tmp_path.iterdir()) == []
    finally:
        os.close(rootfd)


def test_partial_seed_transfer_closes_ambiguous_channel(tmp_path: Path) -> None:
    """An interrupted transfer leaves no file or open reader to admit later."""
    rootfd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    reader, writer = socket.socketpair()
    try:
        write_frame(writer, b"ab", max_bytes=4, deadline=time.monotonic() + 1)
        writer.close()
        with pytest.raises(EOFError):
            receive_storage_seed_files(
                reader,
                manifest={"input": 3},
                expected_api_uid=os.getuid(),
                authority_dirfd=rootfd,
                frame_max_bytes=4,
                max_seed_bytes=8,
                max_seed_entries=1,
                max_manifest_bytes=32,
                deadline=time.monotonic() + 1,
            )
        assert reader.fileno() == -1
        assert list(tmp_path.iterdir()) == []
    finally:
        os.close(rootfd)
        writer.close()


def test_seed_staging_refuses_nonprivate_directory(tmp_path: Path) -> None:
    """Authority file handles must point to a private service-owned directory."""
    os.chmod(tmp_path, 0o755)
    rootfd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    reader, writer = socket.socketpair()
    try:
        with reader, writer, pytest.raises(PermissionError, match="not private"):
            receive_storage_seed_files(
                reader,
                manifest={},
                expected_api_uid=os.getuid(),
                authority_dirfd=rootfd,
                frame_max_bytes=4,
                max_seed_bytes=8,
                max_seed_entries=1,
                max_manifest_bytes=32,
                deadline=time.monotonic() + 1,
            )
    finally:
        os.close(rootfd)


def test_replay_rehashes_staged_wire_content_and_refuses_mutation(tmp_path: Path) -> None:
    """A staged receipt equals the byte path until its file content changes."""
    rootfd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    reader, writer = socket.socketpair()
    try:
        with reader, writer:
            deadline = time.monotonic() + 2
            write_frame(writer, b"abc", max_bytes=4, deadline=deadline)
            with receive_storage_seed_files(
                reader,
                manifest={"input": 3},
                expected_api_uid=os.getuid(),
                authority_dirfd=rootfd,
                frame_max_bytes=4,
                max_seed_bytes=8,
                max_seed_entries=1,
                max_manifest_bytes=32,
                deadline=deadline,
            ) as received:

                def replay(seed: bytes | ReceivedStorageSeedFile) -> StorageAdmissionReplay:
                    return derive_storage_admission_replay(
                        requester=Principal("operator", frozenset({"studio.admin"})),
                        mutation_id="one-mutation",
                        workspace="default",
                        task=resolve_named_studio_task(
                            "analysis.run", authorized_route="/api/analysis/jobs"
                        ),
                        authorized_route="/api/analysis/jobs",
                        payload_json=b'{"model":"lif"}',
                        seed_inputs={"input": seed},
                        execution_timeout_seconds=10.0,
                        queue_wait_seconds=None,
                        admission=None,
                        training_config=None,
                        experiment_sha256=None,
                        max_metadata_bytes=4096,
                        max_seed_bytes=8,
                        max_seed_entries=1,
                    )

                file_replay = replay(received.files[0])
                byte_replay = replay(b"abc")
                assert file_replay.payload_sha256 == byte_replay.payload_sha256
                seed = received.files[0]
                seed.stream.write(b"x")
                seed.stream.flush()
                with pytest.raises(ValueError, match="changed after receipt"):
                    replay(seed)
    finally:
        os.close(rootfd)
