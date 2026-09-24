# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — fixed launcher-started worker entry

"""The fixed worker entry confines itself and runs only a reviewed named task.

The entry changes irreversible process state (no-new-privileges, parent-death
signal, subreaper, resource ceilings), so its ``main`` runs only in child
processes here. Descriptor reading is exercised directly on real spool trees.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time

import pytest

from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_process_protocol import _process_worker_environment
from sc_neurocore.studio.platform.jobs_reaper import reap_process_group
from sc_neurocore.studio.platform.storage_worker_bootstrap import (
    DESCRIPTOR_MAX_BYTES,
    read_worker_descriptor,
)
from sc_neurocore.studio.platform.storage_worker_grant import (
    GRANT_ENDPOINT_NAME,
    ExpectedWorker,
    WorkerGrantEndpoint,
)

_JOB = "sj_" + "c" * 16
_GENERATION = "e" * 32
_LIMITS = {
    "--max-memory-bytes": 4 << 30,
    "--max-cpu-seconds": 600,
    "--max-open-files": 256,
    "--max-file-bytes": 64 << 20,
    # RLIMIT_NPROC counts every process and thread of the UID, not one job;
    # a same-identity run must inherit the account's own ceiling.
    "--max-processes": resource.getrlimit(resource.RLIMIT_NPROC)[0],
}


@dataclass(frozen=True)
class Spool:
    """One API-built generation spool used by a test."""

    root: Path
    generation: Path
    input: Path
    work: Path


def _descriptor(**changes: object) -> dict[str, object]:
    body: dict[str, object] = {
        "version": "studio.worker.descriptor.v1",
        "job_id": _JOB,
        "generation": _GENERATION,
        "task_name": "analysis.run",
        "authorized_route": "/api/analysis/jobs",
        "supervisor": supervisor_identity(),
        "max_artifact_bytes": 1024,
    }
    body.update(changes)
    return body


def _write_descriptor(spool: Spool, content: bytes) -> None:
    path = spool.input / "descriptor.json"
    path.write_bytes(content)
    path.chmod(0o640)


@pytest.fixture
def spool() -> Iterator[Spool]:
    """Build the API side of one generation spool under a short private root."""
    root = Path(tempfile.mkdtemp(prefix="scs"))
    generation = root / _JOB / _GENERATION
    (generation / "input").mkdir(parents=True)
    (generation / _JOB).mkdir()
    for directory in (root / _JOB, generation, generation / "input"):
        directory.chmod(0o750)
    built = Spool(root, generation, generation / "input", generation / _JOB)
    _write_descriptor(built, json.dumps(_descriptor()).encode())
    (built.input / "payload.json").write_text("{}")
    try:
        yield built
    finally:
        shutil.rmtree(root)


def _read(spool: Spool, *, job_id: str = _JOB, generation: str = _GENERATION) -> None:
    read_worker_descriptor(spool.root, job_id=job_id, generation=generation, server_uid=os.getuid())


def test_valid_descriptor_is_read_through_owned_directories(spool: Spool) -> None:
    """The exact API-written descriptor for the launched generation is accepted."""
    descriptor = read_worker_descriptor(
        spool.root, job_id=_JOB, generation=_GENERATION, server_uid=os.getuid()
    )
    assert descriptor.task_name == "analysis.run"
    assert descriptor.job_id == _JOB and descriptor.generation == _GENERATION
    assert descriptor.max_artifact_bytes == 1024


@pytest.mark.parametrize(
    "content,match",
    [
        (b"", "byte limit"),
        (b" " * (DESCRIPTOR_MAX_BYTES + 1), "byte limit"),
        (b"\xff", "invalid worker descriptor JSON"),
        (b"{", "invalid worker descriptor JSON"),
        (b'{"version":"a","version":"a"}', "duplicate worker descriptor field"),
        (json.dumps(_descriptor(job_id="sj_" + "d" * 16)).encode(), "another job generation"),
        (json.dumps(_descriptor(generation="f" * 32)).encode(), "another job generation"),
        (json.dumps(_descriptor(task_name="model.scan")).encode(), "not available on this route"),
        (json.dumps(_descriptor(task_name="unknown")).encode(), "not available on this route"),
        (json.dumps({**_descriptor(), "task_path": "os:system"}).encode(), "Extra inputs"),
        (json.dumps(_descriptor(supervisor="host:0:1")).encode(), "supervisor"),
        (json.dumps(_descriptor(max_artifact_bytes=0)).encode(), "max_artifact_bytes"),
    ],
    # Stable IDs: descriptors embed this process's identity, which differs per
    # collecting interpreter, e.g. per parallel test worker.
    ids=[
        "empty",
        "oversized",
        "not-utf8",
        "truncated",
        "duplicate-field",
        "other-job",
        "other-generation",
        "unreviewed-task",
        "unknown-task",
        "task-path-field",
        "zero-supervisor-pid",
        "zero-artifact-limit",
    ],
)
def test_malformed_or_foreign_descriptor_refuses(spool: Spool, content: bytes, match: str) -> None:
    """Size, encoding, duplicates, foreign generations and unreviewed tasks refuse."""
    _write_descriptor(spool, content)
    with pytest.raises(ValueError, match=match):
        _read(spool)


@pytest.mark.parametrize(
    "job_id,generation",
    [("sj_" + "C" * 16, _GENERATION), ("../" + _JOB, _GENERATION), (_JOB, "e" * 31)],
)
def test_launcher_identifiers_are_checked_before_any_open(
    spool: Spool, job_id: str, generation: str
) -> None:
    """Malformed launcher identifiers cannot traverse the spool."""
    with pytest.raises(ValueError, match="identifier is invalid"):
        _read(spool, job_id=job_id, generation=generation)


def test_relative_spool_root_refuses(spool: Spool) -> None:
    """The spool root must be the configured absolute path."""
    with pytest.raises(ValueError, match="absolute"):
        read_worker_descriptor(
            Path("spool"), job_id=_JOB, generation=_GENERATION, server_uid=os.getuid()
        )


def test_foreign_owner_refuses(spool: Spool) -> None:
    """Directories not owned by the configured API identity are refused."""
    with pytest.raises(PermissionError, match="not owned by the API identity"):
        read_worker_descriptor(
            spool.root, job_id=_JOB, generation=_GENERATION, server_uid=os.getuid() + 1
        )


def test_world_writable_spool_directory_refuses(spool: Spool) -> None:
    """A directory any identity could rewrite is not trusted."""
    spool.generation.chmod(0o777)
    with pytest.raises(PermissionError, match="not owned by the API identity"):
        _read(spool)


@pytest.mark.parametrize("mode", [0o660, 0o646])
def test_group_or_world_writable_descriptor_refuses(spool: Spool, mode: int) -> None:
    """A descriptor that another identity could rewrite is refused."""
    (spool.input / "descriptor.json").chmod(mode)
    with pytest.raises(PermissionError, match="not an API-owned file"):
        _read(spool)


def test_non_regular_descriptor_refuses(spool: Spool) -> None:
    """A directory in place of the descriptor is refused."""
    (spool.input / "descriptor.json").unlink()
    (spool.input / "descriptor.json").mkdir()
    with pytest.raises(PermissionError, match="not an API-owned file"):
        _read(spool)


def test_symbolic_link_component_refuses(spool: Spool) -> None:
    """A symbolic link anywhere in the generation path is never followed."""
    moved = spool.generation.with_name("moved")
    spool.generation.rename(moved)
    spool.generation.symlink_to(moved)
    with pytest.raises(OSError):
        _read(spool)


def test_missing_descriptor_refuses(spool: Spool) -> None:
    """No descriptor means no task selection."""
    (spool.input / "descriptor.json").unlink()
    with pytest.raises(FileNotFoundError):
        _read(spool)


def _bootstrap(spool: Spool, *, launcher_pid: int) -> subprocess.Popen[bytes]:
    command = [
        sys.executable,
        "-m",
        "sc_neurocore.studio.platform.storage_worker_bootstrap",
        "--spool-root",
        str(spool.root),
        "--job",
        _JOB,
        "--generation",
        _GENERATION,
        "--server-uid",
        str(os.getuid()),
        "--launcher-pid",
        str(launcher_pid),
    ]
    for name, value in _LIMITS.items():
        command += [name, str(value)]
    return subprocess.Popen(
        command,
        env=_process_worker_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )


@pytest.mark.parametrize("fault", ["launcher", "descriptor"])
def test_bootstrap_refuses_before_worker_start(spool: Spool, fault: str) -> None:
    """A dead launcher or a missing descriptor stops the entry before any import."""
    if fault == "descriptor":
        (spool.input / "descriptor.json").unlink()
    child = _bootstrap(spool, launcher_pid=1 if fault == "launcher" else os.getpid())
    try:
        _, stderr = child.communicate(timeout=60.0)
    finally:
        assert reap_process_group(child, owned_group_id=child.pid).reaped
    assert child.returncode == 2
    expected = "RuntimeError" if fault == "launcher" else "FileNotFoundError"
    assert stderr.decode().strip() == f"studio worker bootstrap refused: {expected}"
    assert list(spool.work.iterdir()) == []


def _limits(pid: int) -> dict[str, str]:
    rows: dict[str, str] = {}
    with open(f"/proc/{pid}/limits", encoding="ascii") as handle:
        for line in handle.readlines()[1:]:
            rows[line[:26].strip()] = line[26:47].strip()
    return rows


def test_bootstrap_confines_itself_then_runs_the_named_task_after_grant(spool: Spool) -> None:
    """Ceilings and no-new-privileges hold before the grant; the reviewed task then runs."""
    directory = os.open(spool.generation, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    registered: list[str] = []
    try:
        with WorkerGrantEndpoint(directory, spool.generation, GRANT_ENDPOINT_NAME) as endpoint:
            child = _bootstrap(spool, launcher_pid=os.getpid())
            try:
                deadline = time.monotonic() + 60.0
                status = ""
                while "NoNewPrivs:\t1" not in status and time.monotonic() < deadline:
                    with open(f"/proc/{child.pid}/status", encoding="ascii") as handle:
                        status = handle.read()
                    time.sleep(0.02)
                limits = _limits(child.pid)
                while limits["Max open files"] != "256" and time.monotonic() < deadline:
                    time.sleep(0.02)
                    limits = _limits(child.pid)
                assert "NoNewPrivs:\t1" in status
                assert limits["Max address space"] == str(4 << 30)
                assert limits["Max cpu time"] == "600"
                assert limits["Max open files"] == "256"
                assert limits["Max file size"] == str(64 << 20)
                assert limits["Max processes"] == str(_LIMITS["--max-processes"])
                assert limits["Max core file size"] == "0"
                _, pid, token = supervisor_identity(child.pid).split(":", 2)
                endpoint.grant(
                    ExpectedWorker(uid=os.getuid(), pid=int(pid), start_token=token),
                    deadline=time.monotonic() + 60.0,
                    max_refusals=1,
                    register=registered.append,
                )
                _, stderr = child.communicate(timeout=120.0)
            finally:
                assert reap_process_group(child, owned_group_id=child.pid).reaped
    finally:
        os.close(directory)
    assert len(registered) == 1
    assert child.returncode == 1, stderr.decode()
    evidence = json.loads((spool.work / ".studio_process_result.json").read_text())
    assert evidence["status"] == "failed"
    assert evidence["error"] == "AnalysisJobValidationError"
