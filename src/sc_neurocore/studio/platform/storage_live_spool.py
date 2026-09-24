# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — live worker directories of this API generation

"""Live reads and control delivery through the spool of jobs this API supervises.

Live observations stay on the spool (sealed reads come from the authority).
The registry holds a duplicate of each staged worker directory descriptor,
so a read or a delivery never resolves a spool path again. The worker owns
what it writes there: reads never follow a link and accept only regular
files. Control commands and control seeds are written into the directories
the API staged before the worker started, checked to still be the API's own,
under an exclusive temporary name and then renamed into place, as the
embedded manager publishes a command. After a job finishes its directory is
kept for a bounded number of later jobs, so a stream can read its last lines.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import os
import stat
import threading

from sc_neurocore.studio.platform.jobs_models import (
    STUDIO_CONTROL_COMMAND_FILE,
    STUDIO_CONTROL_COMMAND_MAX_BYTES,
    STUDIO_CONTROL_DIR,
    STUDIO_CONTROL_SEED_DIR,
    StudioJobArtifactUnavailable,
    StudioJobRejected,
)
from sc_neurocore.studio.platform.storage_spool_staging import (
    CONTROL_SEED_MODE,
    FILE_MODE,
    canonical_parts,
)

_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_READ = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
_CREATE = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


def _owned_directory(parent: int, name: str, *, create: bool) -> int:
    """Open ``name`` as a directory this identity owns, creating it when asked."""
    created = False
    if create:
        try:
            os.mkdir(name, 0o700, dir_fd=parent)
            created = True
        except FileExistsError:
            pass
    descriptor = os.open(name, _DIRECTORY, dir_fd=parent)
    if os.fstat(descriptor).st_uid != os.geteuid():
        os.close(descriptor)
        raise PermissionError("spool control directory is not owned by the API")
    if created:
        # Explicit, whatever the API's umask: the compute group reads seeds.
        os.fchmod(descriptor, CONTROL_SEED_MODE)
    return descriptor


def _publish(directory: int, name: str, payload: bytes) -> None:
    """Write ``payload`` under an exclusive temporary name, then rename it into place."""
    partial = f".{name}.partial"
    try:
        os.unlink(partial, dir_fd=directory)
    except FileNotFoundError:
        pass
    descriptor = os.open(partial, _CREATE, 0o600, dir_fd=directory)
    try:
        # Explicit, whatever the API's umask: the worker reads what is published.
        os.fchmod(descriptor, FILE_MODE)
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view) :]
    finally:
        os.close(descriptor)
    os.rename(partial, name, src_dir_fd=directory, dst_dir_fd=directory)


class LiveSpools:
    """This API generation's live worker directories, keyed by job."""

    def __init__(self, *, retain: int, max_seed_bytes: int) -> None:
        """Keep ``retain`` finished directories; bound each control seed."""
        if retain < 0 or max_seed_bytes <= 0:
            raise ValueError("live spool bounds must be non-negative and positive")
        self._retain = retain
        self._max_seed_bytes = max_seed_bytes
        self._lock = threading.Lock()
        self._live: dict[str, int] = {}
        self._finished: OrderedDict[str, int] = OrderedDict()

    def attach(self, job_id: str, work: int) -> None:
        """Hold a duplicate of a staged worker directory for ``job_id``."""
        duplicate = os.dup(work)
        with self._lock:
            previous = self._live.pop(job_id, None)
            self._live[job_id] = duplicate
        if previous is not None:
            os.close(previous)

    def retire(self, job_id: str) -> None:
        """Mark an attached ``job_id`` finished; close the oldest beyond the bound.

        Raises
        ------
        KeyError
            The job was not attached.
        """
        closing: list[int] = []
        with self._lock:
            self._finished[job_id] = self._live.pop(job_id)
            while len(self._finished) > self._retain:
                closing.append(self._finished.popitem(last=False)[1])
        for stale in closing:
            os.close(stale)

    def close(self) -> None:
        """Close every held directory."""
        with self._lock:
            held = [*self._live.values(), *self._finished.values()]
            self._live.clear()
            self._finished.clear()
        for descriptor in held:
            os.close(descriptor)

    def _held(self, job_id: str) -> int | None:
        with self._lock:
            descriptor = self._live.get(job_id, self._finished.get(job_id))
            return None if descriptor is None else os.dup(descriptor)

    def read(
        self, job_id: str, relative_path: str, *, offset: int, max_bytes: int
    ) -> tuple[bytes, int]:
        """Return up to ``max_bytes`` appended after ``offset`` and the new offset.

        A job this generation does not hold, or an entry not yet written,
        yields nothing and the same offset, as the embedded manager does.

        Raises
        ------
        ValueError
            The offset is negative or the size not positive.
        StudioJobArtifactUnavailable
            The path escapes or the entry is a link or not a regular file.
        """
        if offset < 0:
            raise ValueError("Studio live artifact offset must be non-negative.")
        if max_bytes <= 0:
            raise ValueError("Studio live artifact read size must be positive.")
        try:
            *directories, name = canonical_parts(relative_path)
        except ValueError as exc:
            raise StudioJobArtifactUnavailable(str(exc)) from exc
        work = self._held(job_id)
        if work is None:
            return b"", offset
        held = [work]
        try:
            for directory in directories:
                held.append(os.open(directory, _DIRECTORY, dir_fd=held[-1]))
            handle = os.open(name, _READ, dir_fd=held[-1])
        except FileNotFoundError:
            return b"", offset
        except OSError as exc:
            raise StudioJobArtifactUnavailable("Studio live artifact is unavailable.") from exc
        finally:
            for descriptor in reversed(held):
                os.close(descriptor)
        try:
            if not stat.S_ISREG(os.fstat(handle).st_mode):
                raise StudioJobArtifactUnavailable("Studio live artifact is unavailable.")
            payload = os.pread(handle, max_bytes, offset)
        finally:
            os.close(handle)
        return payload, offset + len(payload)

    def deliver(self, job_id: str, command: bytes, seeds: Mapping[str, bytes]) -> None:
        """Publish control seeds, then the command, into the job's live spool.

        Raises
        ------
        StudioJobRejected
            This generation holds no live directory for the job, a limit is
            exceeded, or a seed path is not canonical.
        PermissionError
            A control directory is no longer the API's own.
        """
        if len(command) > STUDIO_CONTROL_COMMAND_MAX_BYTES:
            raise StudioJobRejected("Studio job control command exceeds configured size limit.")
        if any(len(payload) > self._max_seed_bytes for payload in seeds.values()):
            raise StudioJobRejected("Studio job seed input exceeds configured size limit.")
        try:
            parts = {path: canonical_parts(path) for path in seeds}
        except ValueError as exc:
            raise StudioJobRejected(str(exc)) from exc
        with self._lock:
            live = self._live.get(job_id)
            work = None if live is None else os.dup(live)
        if work is None:
            raise StudioJobRejected("Studio job work directory is unavailable.")
        try:
            for path, payload in seeds.items():
                *directories, name = parts[path]
                parent = _owned_directory(work, STUDIO_CONTROL_SEED_DIR, create=False)
                try:
                    for directory in directories:
                        child = _owned_directory(parent, directory, create=True)
                        os.close(parent)
                        parent = child
                    _publish(parent, name, bytes(payload))
                finally:
                    os.close(parent)
            control = _owned_directory(work, STUDIO_CONTROL_DIR, create=False)
            try:
                _publish(control, STUDIO_CONTROL_COMMAND_FILE, command)
            finally:
                os.close(control)
        finally:
            os.close(work)


__all__ = ["LiveSpools"]
