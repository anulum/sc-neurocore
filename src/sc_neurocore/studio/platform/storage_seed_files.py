# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — file-backed storage seed ingress

"""Receive verified seed frames into private, unlinked service-owned files."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import os
import socket
import stat
from tempfile import TemporaryFile
from typing import BinaryIO

from sc_neurocore.studio.platform.storage_peer import read_verified_frame, require_storage_peer
from sc_neurocore.studio.platform.storage_seed_ingress import validate_storage_seed_manifest


@dataclass(frozen=True, slots=True)
class ReceivedStorageSeedFile:
    """One logical seed and the digest of bytes actually received from the wire."""

    name: str
    stream: BinaryIO
    size: int
    sha256: str


class ReceivedStorageSeedFiles:
    """Own all private seed handles until admission transfers or rejects them."""

    def __init__(self, files: tuple[ReceivedStorageSeedFile, ...]) -> None:
        self.files = files

    def close(self) -> None:
        """Release every unlinked seed file, including after a handler error."""
        for seed in self.files:
            seed.stream.close()

    def __enter__(self) -> ReceivedStorageSeedFiles:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def receive_storage_seed_files(
    channel: socket.socket,
    *,
    manifest: Mapping[str, int],
    expected_api_uid: int,
    authority_dirfd: int,
    frame_max_bytes: int,
    max_seed_bytes: int,
    max_seed_entries: int,
    max_manifest_bytes: int,
    deadline: float,
) -> ReceivedStorageSeedFiles:
    """Stream one bounded transfer into unlinked files under a held authority.

    The caller owns the returned handles and must close them after the
    admission handler has transferred the inputs. The authority directory
    descriptor remains owned by the caller. No request path becomes a disk
    path; each temporary file is private to the storage service UID.
    A failed or ambiguous transfer closes the channel and every staged file.
    """
    if type(expected_api_uid) is not int or not 0 <= expected_api_uid < 0xFFFFFFFF:
        raise ValueError("invalid storage API UID")
    if type(authority_dirfd) is not int or authority_dirfd < 0:
        raise ValueError("invalid storage authority directory")
    if not isinstance(manifest, Mapping):
        raise ValueError("invalid storage seed manifest")
    authority = os.fstat(authority_dirfd)
    if (
        not stat.S_ISDIR(authority.st_mode)
        or authority.st_uid != os.geteuid()
        or stat.S_IMODE(authority.st_mode) != 0o700
    ):
        raise PermissionError("storage authority directory is not private")
    declarations = dict(manifest.items())
    names = validate_storage_seed_manifest(
        declarations,
        frame_max_bytes=frame_max_bytes,
        max_seed_bytes=max_seed_bytes,
        max_seed_entries=max_seed_entries,
        max_manifest_bytes=max_manifest_bytes,
        deadline=deadline,
    )
    staged: list[ReceivedStorageSeedFile] = []
    try:
        require_storage_peer(channel, expected_uid=expected_api_uid)
        for name in names:
            size = declarations[name]
            # Ownership transfers to ReceivedStorageSeedFiles on success.
            stream = TemporaryFile(  # noqa: SIM115
                mode="w+b", dir=f"/proc/self/fd/{authority_dirfd}"
            )
            try:
                metadata = os.fstat(stream.fileno())
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != authority.st_uid
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                    or metadata.st_dev != authority.st_dev
                ):
                    raise PermissionError("storage seed staging file is not private")
                digest = hashlib.sha256()
                received = 0
                while received < size:
                    chunk = read_verified_frame(
                        channel,
                        expected_uid=expected_api_uid,
                        max_bytes=min(frame_max_bytes, size - received),
                        deadline=deadline,
                    )
                    digest.update(chunk)
                    view = memoryview(chunk)
                    while view:
                        written = stream.write(view)
                        if written is None or written <= 0:
                            raise OSError("storage seed staging write failed")
                        view = view[written:]
                    received += len(chunk)
                stream.flush()
                stream.seek(0)
                staged.append(
                    ReceivedStorageSeedFile(
                        name=name,
                        stream=stream,
                        size=size,
                        sha256=digest.hexdigest(),
                    )
                )
            except BaseException:
                stream.close()
                raise
        return ReceivedStorageSeedFiles(tuple(staged))
    except BaseException:
        for seed in staged:
            seed.stream.close()
        channel.close()
        raise
