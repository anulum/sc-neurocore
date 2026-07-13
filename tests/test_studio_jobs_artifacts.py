# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio job sandbox contract tests

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")


import sc_neurocore.studio.platform.jobs as jobs_module
from sc_neurocore.studio.platform.jobs import (
    StudioJobContext,
    StudioJobManager,
    StudioJobRejected,
)


def test_studio_job_manager_completes_job_with_path_free_artifact_manifest(
    tmp_path: Path,
) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=1.0,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("reports/result.txt", b"ok")
        return {"ok": True}

    record = manager.submit(
        kind="synthesis",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    completed = manager.wait(record.job_id, timeout_seconds=2.0)

    assert completed.status == "completed"
    assert completed.result == {"ok": True}
    assert len(completed.artifacts) == 1
    assert completed.artifacts[0].relative_path == "reports/result.txt"
    assert manager.list_snapshot().to_public_dict()["schema_version"] == "studio.jobs.list.v1"
    assert str(tmp_path) not in str(completed.to_public_dict())
    assert (tmp_path / "jobs" / record.job_id / "reports" / "result.txt").read_bytes() == b"ok"


def test_studio_job_manager_reads_manifest_declared_artifacts(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=1.0,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("reports/result.txt", "artifact body")
        return {"ok": True}

    record = manager.submit(
        kind="synthesis",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    manager.wait(record.job_id, timeout_seconds=2.0)

    payload = manager.read_artifact(record.job_id, "reports/result.txt")

    assert payload.payload == b"artifact body"
    assert payload.artifact.relative_path == "reports/result.txt"
    assert payload.artifact.sha256 == hashlib.sha256(b"artifact body").hexdigest()


def test_studio_job_manager_purges_terminal_record_and_directory(tmp_path: Path) -> None:
    """Terminal job purges remove the record and confined work directory."""

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=1.0,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("reports/result.txt", "artifact body")
        return {"ok": True}

    record = manager.submit(
        kind="synthesis",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    completed = manager.wait(record.job_id, timeout_seconds=2.0)
    job_dir = tmp_path / "jobs" / record.job_id

    purged = manager.purge_terminal_record(record.job_id)

    assert completed.status == "completed"
    assert purged.job_id == record.job_id
    assert manager.list_records() == ()
    assert not job_dir.exists()
    with pytest.raises(KeyError):
        manager.record(record.job_id)
    with pytest.raises(KeyError):
        manager.read_artifact(record.job_id, "reports/result.txt")


def test_studio_job_manager_rejects_active_record_purge(tmp_path: Path) -> None:
    """Active jobs cannot be purged while their worker may still write files."""

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=2.0,
    )
    release = threading.Event()

    def task(context: StudioJobContext) -> dict[str, object]:
        release.wait(timeout=1.0)
        context.write_artifact("reports/result.txt", "artifact body")
        return {"ok": True}

    record = manager.submit(
        kind="training",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )

    with pytest.raises(StudioJobRejected, match="active"):
        manager.purge_terminal_record(record.job_id)

    release.set()
    completed = manager.wait(record.job_id, timeout_seconds=3.0)

    assert completed.status == "completed"
    assert manager.record(record.job_id).job_id == record.job_id


def test_studio_job_manager_rejects_non_directory_purge_target(tmp_path: Path) -> None:
    """Job purges reject corrupted non-directory job targets."""

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=1.0,
    )

    def task(_context: StudioJobContext) -> dict[str, object]:
        return {"ok": True}

    record = manager.submit(
        kind="synthesis",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    completed = manager.wait(record.job_id, timeout_seconds=2.0)
    job_dir = tmp_path / "jobs" / record.job_id
    job_dir.rmdir()
    job_dir.write_text("not a directory", encoding="utf-8")

    with pytest.raises(StudioJobRejected, match="not a directory"):
        manager.purge_terminal_record(record.job_id)

    assert completed.status == "completed"
    assert manager.record(record.job_id).job_id == record.job_id


def test_studio_job_directory_resolution_rejects_non_generated_ids(tmp_path: Path) -> None:
    """Job directory lookup accepts only generated job identifier shapes."""

    with pytest.raises(ValueError, match="escapes the job root"):
        jobs_module._resolve_job_directory(
            root=tmp_path / "jobs",
            job_id="../escape",
            error_message="Studio job path escapes the job root.",
        )
    with pytest.raises(ValueError, match="escapes the job root"):
        jobs_module._resolve_job_directory(
            root=tmp_path / "jobs",
            job_id="sj_not_hex",
            error_message="Studio job path escapes the job root.",
        )
