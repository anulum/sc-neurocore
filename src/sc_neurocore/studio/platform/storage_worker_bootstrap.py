# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — fixed launcher-started worker entry

"""Fixed entry that a launcher runs under the compute identity for one generation.

The launcher supplies only its own configuration on the command line: spool
root, job ID, generation, API identity, its own PID and resource ceilings.
Before reading any request data the entry forbids privilege gain, dies with
its launcher, becomes a child subreaper, sets a group-readable umask and
applies the ceilings. It then
reads a bounded descriptor written by the trusted API into the generation's
input directory, resolves the task name through the reviewed catalogue and
runs the existing worker, which imports nothing until the API grant arrives.

Spool layout for one generation, all directories owned by the API identity::

    <spool_root>/<job_id>/<generation>/
        input/descriptor.json      API-written, worker-readable
        input/payload.json         API-written, worker-readable
        grant.sock                 API grant endpoint
        <job_id>/                  worker directory; seeds and control inside

The worker directory is named by the job ID because tasks read
``StudioJobContext.job_id`` from it.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import resource
import signal
import stat
import sys
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from sc_neurocore.studio.platform.storage_named_tasks import resolve_named_studio_task
from sc_neurocore.studio.platform.storage_worker_grant import GRANT_ENDPOINT_NAME
from sc_neurocore.studio.platform.storage_worker_tree import (
    become_child_subreaper,
    process_control,
)

DESCRIPTOR_VERSION = "studio.worker.descriptor.v1"
DESCRIPTOR_MAX_BYTES = 4096
_PR_SET_PDEATHSIG = 1
_PR_SET_NO_NEW_PRIVS = 38
_JOB_ID = re.compile(r"sj_[0-9a-f]{16}")
_GENERATION = re.compile(r"[0-9a-f]{32}")


class WorkerDescriptor(BaseModel):
    """API-written selection for one generation; the launcher never reads it.

    ``supervisor`` is the API process generation the worker guard follows.
    The task is selected by reviewed name and route, never by import path.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    version: Literal["studio.worker.descriptor.v1"]
    job_id: Annotated[str, Field(pattern=r"^sj_[0-9a-f]{16}$")]
    generation: Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
    task_name: Annotated[str, Field(min_length=1, max_length=128)]
    authorized_route: Annotated[str, Field(min_length=1, max_length=256)]
    supervisor: Annotated[str, Field(pattern=r"^[^:\s]{1,253}:[1-9][0-9]{0,9}:[1-9][0-9]{0,19}$")]
    max_artifact_bytes: Annotated[int, Field(gt=0, le=1 << 40)]


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    fields: dict[str, object] = {}
    for name, value in pairs:
        if name in fields:
            raise ValueError("duplicate worker descriptor field")
        fields[name] = value
    return fields


def _open_owned_directory(parent: int, name: str, owner: int) -> int:
    descriptor = os.open(
        name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent
    )
    metadata = os.fstat(descriptor)
    if metadata.st_uid != owner or metadata.st_mode & 0o002:
        os.close(descriptor)
        raise PermissionError("worker spool directory is not owned by the API identity")
    return descriptor


def read_worker_descriptor(
    spool_root: Path, *, job_id: str, generation: str, server_uid: int
) -> WorkerDescriptor:
    """Read and check the API-written descriptor for one exact generation.

    Parameters
    ----------
    spool_root : Path
        Absolute compute spool root from the launcher configuration.
    job_id, generation : str
        Launcher-supplied identifiers; the descriptor must repeat them.
    server_uid : int
        Configured API identity that must own every spool directory used.

    Returns
    -------
    WorkerDescriptor
        Strictly validated descriptor whose task pairing is in the catalogue.

    Raises
    ------
    ValueError
        Identifiers, size, JSON, schema or task pairing is invalid.
    PermissionError
        A spool directory or the descriptor is not API-owned or is writable by
        others, or a path component is a symbolic link.
    OSError
        A spool entry is missing.
    """
    if _JOB_ID.fullmatch(job_id) is None or _GENERATION.fullmatch(generation) is None:
        raise ValueError("worker job or generation identifier is invalid")
    if not spool_root.is_absolute():
        raise ValueError("worker spool root must be absolute")
    opened: list[int] = []
    try:
        opened.append(
            os.open(spool_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        )
        for name in (job_id, generation, "input"):
            opened.append(_open_owned_directory(opened[-1], name, server_uid))
        handle = os.open(
            "descriptor.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=opened[-1]
        )
        try:
            metadata = os.fstat(handle)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != server_uid
                or metadata.st_mode & 0o022
            ):
                raise PermissionError("worker descriptor is not an API-owned file")
            raw = os.read(handle, DESCRIPTOR_MAX_BYTES + 1)
        finally:
            os.close(handle)
    finally:
        for directory in reversed(opened):
            os.close(directory)
    if not 0 < len(raw) <= DESCRIPTOR_MAX_BYTES:
        raise ValueError("worker descriptor exceeds byte limit")
    try:
        text = raw.decode("utf-8")
        json.loads(text, object_pairs_hook=_unique_fields)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("invalid worker descriptor JSON") from exc
    descriptor = WorkerDescriptor.model_validate_json(text, strict=True)
    if descriptor.job_id != job_id or descriptor.generation != generation:
        raise ValueError("worker descriptor names another job generation")
    resolve_named_studio_task(descriptor.task_name, authorized_route=descriptor.authorized_route)
    return descriptor


def _apply_limits(args: argparse.Namespace) -> None:
    for limit, value in (
        (resource.RLIMIT_AS, args.max_memory_bytes),
        (resource.RLIMIT_CPU, args.max_cpu_seconds),
        (resource.RLIMIT_NOFILE, args.max_open_files),
        (resource.RLIMIT_FSIZE, args.max_file_bytes),
        (resource.RLIMIT_NPROC, args.max_processes),
        (resource.RLIMIT_CORE, 0),
    ):
        resource.setrlimit(limit, (value, value))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="studio-worker-bootstrap")
    parser.add_argument("--spool-root", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--server-uid", type=int, required=True)
    parser.add_argument("--launcher-pid", type=int, required=True)
    for name in (
        "--max-memory-bytes",
        "--max-cpu-seconds",
        "--max-open-files",
        "--max-file-bytes",
        "--max-processes",
    ):
        parser.add_argument(name, type=int, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Confine this process, read its descriptor and run the gated worker.

    Parameters
    ----------
    argv : Sequence[str] or None
        Launcher-built argument vector; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        The existing worker's exit status, or ``2`` when confinement or the
        descriptor refuses before the worker starts. No task is imported on
        any refusal path.
    """
    args = _parse_args(argv)
    try:
        process_control(_PR_SET_NO_NEW_PRIVS, 1)
        process_control(_PR_SET_PDEATHSIG, signal.SIGKILL)
        if os.getppid() != args.launcher_pid:
            raise RuntimeError("worker launcher exited before confinement")
        become_child_subreaper()
        # Everything the worker creates stays readable by the compute group,
        # which the API belongs to, and closed to others.
        os.umask(0o027)
        _apply_limits(args)
        spool_root = Path(args.spool_root)
        descriptor = read_worker_descriptor(
            spool_root, job_id=args.job, generation=args.generation, server_uid=args.server_uid
        )
        task = resolve_named_studio_task(
            descriptor.task_name, authorized_route=descriptor.authorized_route
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"studio worker bootstrap refused: {type(exc).__name__}", file=sys.stderr)
        return 2
    generation_dir = spool_root / args.job / args.generation
    work_dir = generation_dir / args.job
    from sc_neurocore.studio.platform import process_worker

    return process_worker.main(
        [
            "--task",
            task.task_path,
            "--payload",
            str(generation_dir / "input" / "payload.json"),
            "--result",
            str(work_dir / ".studio_process_result.json"),
            "--work-dir",
            str(work_dir),
            "--max-artifact-bytes",
            str(descriptor.max_artifact_bytes),
            "--supervisor",
            descriptor.supervisor,
            "--grant-socket",
            str(generation_dir / GRANT_ENDPOINT_NAME),
            "--grant-server-uid",
            str(args.server_uid),
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
