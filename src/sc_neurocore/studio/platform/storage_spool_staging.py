# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — API staging of one launch generation's compute spool

"""Stage the compute spool of one launch generation from the trusted API.

Every directory is created by the API identity and group-owned by the compute
group, so the launched worker can traverse and read its inputs but can rename
or replace nothing the API wrote. Only the worker directory is group-writable
(set-group-ID, so entries the worker creates stay in that group). Every step
goes through held directory descriptors and never follows a link; an entry
that already exists is refused, never adopted, except a job directory that an
earlier generation of the same job left with exactly the expected ownership.

Layout, matching :mod:`storage_worker_bootstrap`::

    <spool_root>/<job_id>/<generation>/
        input/descriptor.json
        input/payload.json
        <job_id>/                  worker directory
        <job_id>/.studio_seed/...  submission seeds, read-only for the worker
        <job_id>/.studio_control/  control commands; the worker removes each
        <job_id>/.studio_control_seed/  control seeds, read-only for the worker

The control directories exist before the worker starts, so every later API
write goes into a directory the API created and still owns.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
from types import TracebackType

from sc_neurocore.studio.platform.jobs_models import (
    STUDIO_CONTROL_DIR,
    STUDIO_CONTROL_SEED_DIR,
    STUDIO_SEED_INPUT_DIR,
)
from sc_neurocore.studio.platform.jobs_paths import _relative_path_candidate
from sc_neurocore.studio.platform.storage_worker_bootstrap import WorkerDescriptor

DIRECTORY_MODE = 0o750
WORK_DIRECTORY_MODE = 0o2770
# Control seeds are written later by the API; set-group-ID keeps them readable
# by the compute group without changing ownership after the fact.
CONTROL_SEED_MODE = 0o2750
FILE_MODE = 0o640
_DIRECTORY = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_CREATE = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


@dataclass(frozen=True, slots=True)
class StagedGeneration:
    """Held descriptors of one staged generation and its worker directory.

    The caller owns both descriptors and closes them with :meth:`close`; they
    are never passed to a worker.
    """

    path: Path
    directory: int
    work: int

    def close(self) -> None:
        """Close both held descriptors."""
        os.close(self.work)
        os.close(self.directory)

    def __enter__(self) -> StagedGeneration:
        """Return this staged generation."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the descriptors without suppressing caller failures."""
        self.close()


def _expected(metadata: os.stat_result, *, group: int, mode: int) -> bool:
    return (
        stat.S_ISDIR(metadata.st_mode)
        and metadata.st_uid == os.geteuid()
        and metadata.st_gid == group
        and stat.S_IMODE(metadata.st_mode) == mode
    )


def _directory(parent: int, name: str, *, group: int, mode: int, reuse: bool = False) -> int:
    """Create ``name`` under ``parent`` with exact group and mode; return it held.

    With ``reuse`` an existing directory is accepted only when it already has
    exactly this ownership and mode. A created directory is checked as well, so
    a filesystem that does not keep the requested group or mode is refused.
    """
    try:
        os.mkdir(name, 0o700, dir_fd=parent)
        created = True
    except FileExistsError:
        if not reuse:
            raise
        created = False
    descriptor = os.open(name, _DIRECTORY, dir_fd=parent)
    try:
        if created:
            os.fchown(descriptor, -1, group)
            os.fchmod(descriptor, mode)
        if not _expected(os.fstat(descriptor), group=group, mode=mode):
            raise PermissionError("spool directory has unexpected ownership or mode")
    except BaseException:
        os.close(descriptor)
        if created:
            # Still empty: nothing was created inside before its setup failed.
            os.rmdir(name, dir_fd=parent)
        raise
    return descriptor


def _write(parent: int, name: str, payload: bytes, *, group: int) -> None:
    descriptor = os.open(name, _CREATE, FILE_MODE, dir_fd=parent)
    try:
        os.fchown(descriptor, -1, group)
        os.fchmod(descriptor, FILE_MODE)
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view) :]
    finally:
        os.close(descriptor)


def canonical_parts(relative_path: str) -> tuple[str, ...]:
    """Return the components of a printable, canonical, confined relative path.

    Raises
    ------
    ValueError
        The path is not text, not printable, escapes, or is not canonical.
    """
    if not isinstance(relative_path, str) or not relative_path.isprintable():
        raise ValueError("spool path must be printable text")
    candidate = _relative_path_candidate(
        relative_path, error_message="spool path escapes its directory"
    )
    if candidate.as_posix() != relative_path:
        raise ValueError("spool path must be canonical")
    return candidate.parts


def _stage_seeds(work: int, seeds: Mapping[str, bytes], *, group: int) -> None:
    if not seeds:
        return
    seed_root = _directory(work, STUDIO_SEED_INPUT_DIR, group=group, mode=DIRECTORY_MODE)
    try:
        for relative_path, payload in seeds.items():
            *directories, name = canonical_parts(relative_path)
            parent = os.dup(seed_root)
            try:
                for directory in directories:
                    child = _directory(
                        parent, directory, group=group, mode=DIRECTORY_MODE, reuse=True
                    )
                    os.close(parent)
                    parent = child
                _write(parent, name, bytes(payload), group=group)
            finally:
                os.close(parent)
    finally:
        os.close(seed_root)


def stage_generation(
    spool_root: Path,
    descriptor: WorkerDescriptor,
    *,
    payload: bytes,
    seeds: Mapping[str, bytes],
    group: int,
) -> StagedGeneration:
    """Create one generation's inputs and worker directory in the compute spool.

    Parameters
    ----------
    spool_root : Path
        Absolute compute spool root from trusted configuration.
    descriptor : WorkerDescriptor
        Validated descriptor naming the job, generation and task.
    payload : bytes
        Task payload JSON already validated for this job.
    seeds : Mapping[str, bytes]
        Submission seeds by canonical relative path.
    group : int
        Compute group that must read the inputs and write the worker directory.

    Returns
    -------
    StagedGeneration
        Held generation and worker directories.

    Raises
    ------
    ValueError
        The spool root is relative or a seed path is not canonical.
    PermissionError
        A reused job directory has other ownership or mode.
    FileExistsError
        The generation, or any entry inside it, already exists.
    OSError
        A directory or file cannot be created, owned or written.
    """
    if not spool_root.is_absolute():
        raise ValueError("spool root must be absolute")
    # Every descriptor field is length-bounded, so the encoding stays far below
    # DESCRIPTOR_MAX_BYTES, which the bootstrap enforces again when reading.
    encoded = json.dumps(descriptor.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    job_id = descriptor.job_id
    passing: list[int] = []
    kept: list[int] = []
    try:
        passing.append(os.open(spool_root, _DIRECTORY))
        passing.append(
            _directory(passing[-1], job_id, group=group, mode=DIRECTORY_MODE, reuse=True)
        )
        kept.append(
            _directory(passing[-1], descriptor.generation, group=group, mode=DIRECTORY_MODE)
        )
        passing.append(_directory(kept[0], "input", group=group, mode=DIRECTORY_MODE))
        _write(passing[-1], "descriptor.json", encoded, group=group)
        _write(passing[-1], "payload.json", payload, group=group)
        kept.append(_directory(kept[0], job_id, group=group, mode=WORK_DIRECTORY_MODE))
        _stage_seeds(kept[1], seeds, group=group)
        for name, mode in (
            (STUDIO_CONTROL_DIR, WORK_DIRECTORY_MODE),
            (STUDIO_CONTROL_SEED_DIR, CONTROL_SEED_MODE),
        ):
            os.close(_directory(kept[1], name, group=group, mode=mode))
    except BaseException:
        for held in reversed(kept):
            os.close(held)
        raise
    finally:
        for held in reversed(passing):
            os.close(held)
    return StagedGeneration(
        path=spool_root / job_id / descriptor.generation, directory=kept[0], work=kept[1]
    )


__all__ = [
    "CONTROL_SEED_MODE",
    "canonical_parts",
    "DIRECTORY_MODE",
    "FILE_MODE",
    "WORK_DIRECTORY_MODE",
    "StagedGeneration",
    "stage_generation",
]
