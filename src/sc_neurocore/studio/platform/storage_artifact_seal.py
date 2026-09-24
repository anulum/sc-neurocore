# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — authority-side artefact sealing

"""Write verified artefact bytes into the authority's private job directory.

Sealed artefacts live at ``<authority root>/<job id>/<relative path>``, the
layout the job manager already reads completed artefacts from. Every directory
is opened relative to a held descriptor without following symbolic links. A
file is written under a private partial name, synchronised, made read-only and
renamed into place without replacing anything. An existing final file is
accepted only when its bytes are identical, so a retry after a crash between
sealing and the ledger commit completes instead of failing.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
from types import TracebackType

from sc_neurocore.studio.platform.jobs_purge_paths import move_without_replace

_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def _private(metadata: os.stat_result) -> bool:
    return metadata.st_uid == os.geteuid() and not stat.S_IMODE(metadata.st_mode) & 0o077


def _open_directory(name: str, parent: int) -> int:
    """Open or create one private child directory without following links."""
    try:
        os.mkdir(name, 0o700, dir_fd=parent)
    except FileExistsError:
        pass
    descriptor = os.open(name, _DIRECTORY, dir_fd=parent)
    if not _private(os.fstat(descriptor)):
        os.close(descriptor)
        raise PermissionError("sealed artefact directory is not private")
    return descriptor


def _matches(name: str, parent: int, *, size: int, sha256: str) -> bool:
    """Return whether an existing final entry is a regular file with these bytes."""
    try:
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
    except OSError:
        return False
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != size:
            return False
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
        return digest.hexdigest() == sha256
    finally:
        os.close(descriptor)


class SealedArtifactWriter:
    """Seal one job's artefacts under a private authority root.

    Parameters
    ----------
    root : Path
        Authority root that holds the job ledger; it must be private to this
        identity.
    job_id : str
        Validated job identifier naming the job directory.
    """

    def __init__(self, root: Path, job_id: str) -> None:
        """Hold the authority root and the job directory, creating it if needed."""
        self._root = os.open(root, _DIRECTORY)
        try:
            if not _private(os.fstat(self._root)):
                raise PermissionError("storage authority root is not private")
            self._job = _open_directory(job_id, self._root)
        except BaseException:
            os.close(self._root)
            raise

    def seal(self, relative_path: str, payload: bytes, *, sha256: str) -> None:
        """Seal one verified artefact at its canonical job-relative path.

        Parameters
        ----------
        relative_path : str
            Canonical path already validated by the finish protocol.
        payload : bytes
            Complete artefact bytes whose digest the caller verified.
        sha256 : str
            That digest, used to accept an identical existing file.

        Raises
        ------
        FileExistsError
            A different entry already holds the final path.
        PermissionError
            An intermediate directory is not private to this identity.
        """
        *directories, name = relative_path.split("/")
        parents = [self._job]
        try:
            for directory in directories:
                parents.append(_open_directory(directory, parents[-1]))
            parent = parents[-1]
            partial = f".{name}.partial"
            try:
                os.unlink(partial, dir_fd=parent)
            except FileNotFoundError:
                pass
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
            descriptor = os.open(partial, flags, 0o600, dir_fd=parent)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                os.fchmod(handle.fileno(), 0o400)
            directory_path = Path(f"/proc/self/fd/{parent}")
            if not move_without_replace(directory_path / partial, directory_path / name):
                os.unlink(partial, dir_fd=parent)
                if not _matches(name, parent, size=len(payload), sha256=sha256):
                    raise FileExistsError("a different artefact already holds this path")
            os.fsync(parent)
        finally:
            for descriptor in parents[1:]:
                os.close(descriptor)

    def close(self) -> None:
        """Synchronise and release the held job and root directories."""
        os.fsync(self._job)
        os.close(self._job)
        os.close(self._root)

    def __enter__(self) -> SealedArtifactWriter:
        """Return this writer for one finish request."""
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release the held directories on every outcome."""
        self.close()
