# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio jobs suite sentinel

"""Sentinel and adversarial edges for the split Studio jobs test suite."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import cast

import pytest

import sc_neurocore.studio.platform.jobs as jobs_module
from sc_neurocore.studio.platform import jobs
from sc_neurocore.studio.platform.jobs import (
    STUDIO_CONTROL_COMMAND_FILE,
    STUDIO_CONTROL_DIR,
    STUDIO_CONTROL_SEED_DIR,
    STUDIO_SEED_INPUT_DIR,
    StudioJobArtifact,
    StudioJobArtifactUnavailable,
    StudioJobContext,
    StudioJobManager,
    StudioJobRecord,
    StudioJobRejected,
)
from tests.studio_jobs_support import EXPECTED_JOBS_EXPORTS


def test_studio_jobs_facade_exports_are_unique_and_complete() -> None:
    """Keep the historical facade explicit while behavior lives in focused suites."""

    assert tuple(jobs.__all__) == EXPECTED_JOBS_EXPORTS
    assert len(jobs.__all__) == len(set(jobs.__all__)) == 18


def _manager(tmp_path: Path) -> StudioJobManager:
    return StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=1.0,
    )


def _record(
    job_id: str,
    *,
    status: jobs.StudioJobStatus = "completed",
    artifacts: tuple[StudioJobArtifact, ...] = (),
) -> StudioJobRecord:
    return StudioJobRecord(
        job_id=job_id,
        kind="synthesis",
        owner="operator",
        request_id=None,
        status=status,
        execution_model="process",
        created_at_utc="2026-07-13T00:00:00Z",
        artifacts=artifacts,
    )


def _insert_record(manager: StudioJobManager, key: str, record: StudioJobRecord) -> None:
    manager._records[key] = record
    manager._done_events[key] = threading.Event()
    manager._cancel_events[key] = threading.Event()


def test_studio_job_context_validates_seed_size_and_malformed_controls(
    tmp_path: Path,
) -> None:
    """Cover successful and oversized seed reads plus malformed control payloads."""

    work_dir = tmp_path / "work"
    seed_path = work_dir / STUDIO_SEED_INPUT_DIR / "seed.bin"
    control_seed_path = work_dir / STUDIO_CONTROL_SEED_DIR / "seed.bin"
    command_path = work_dir / STUDIO_CONTROL_DIR / STUDIO_CONTROL_COMMAND_FILE
    for path in (seed_path, control_seed_path, command_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_bytes(b"ok")
    control_seed_path.write_bytes(b"ok")
    context = StudioJobContext(
        job_id="sj_0000000000000000",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=2,
    )
    assert context.read_seed_input("seed.bin") == b"ok"
    assert context.read_control_seed("seed.bin") == b"ok"
    seed_path.write_bytes(b"big")
    control_seed_path.write_bytes(b"big")
    with pytest.raises(ValueError, match="seed input exceeds"):
        context.read_seed_input("seed.bin")
    with pytest.raises(ValueError, match="control seed exceeds"):
        context.read_control_seed("seed.bin")
    command_path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="not valid JSON"):
        context.poll_control_command()
    command_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a JSON object"):
        context.poll_control_command()


def test_studio_job_manager_rejects_corrupt_record_paths_and_live_symlinks(
    tmp_path: Path,
) -> None:
    """Translate corrupt stored paths without allowing purge or artifact escape."""

    manager = _manager(tmp_path)
    artifact = StudioJobArtifact(relative_path="artifact.bin", size_bytes=1, sha256="0")
    _insert_record(manager, "bad", _record("../escape", artifacts=(artifact,)))
    with pytest.raises(StudioJobRejected, match="escapes"):
        manager.purge_terminal_record("bad")
    with pytest.raises(StudioJobArtifactUnavailable, match="escapes"):
        manager.read_artifact("bad", "artifact.bin")

    absent_id = "sj_0000000000000000"
    _insert_record(manager, absent_id, _record(absent_id))
    assert manager.purge_terminal_record(absent_id).job_id == absent_id

    live_id = "sj_1111111111111111"
    _insert_record(manager, live_id, _record(live_id, status="running"))
    work_dir = tmp_path / "jobs" / live_id
    outside = tmp_path / "outside"
    work_dir.mkdir(parents=True)
    outside.mkdir()
    (work_dir / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(StudioJobArtifactUnavailable, match="escapes"):
        manager.read_live_artifact_bytes("sj_1111111111111111", "link/events.jsonl", offset=0)


def test_studio_job_manager_rejects_invalid_process_state_and_seed_types(
    tmp_path: Path,
) -> None:
    """Reject corrupt process paths, absent workdirs, and invalid seed boundaries."""

    manager = _manager(tmp_path)
    _insert_record(manager, "bad", _record("../escape", status="running"))
    with pytest.raises(StudioJobRejected, match="escapes"):
        manager.send_control_command("bad", command={"action": "stop"})

    missing_id = "sj_2222222222222222"
    _insert_record(manager, missing_id, _record(missing_id, status="running"))
    with pytest.raises(StudioJobRejected, match="work directory is unavailable"):
        manager.send_control_command(missing_id, command={"action": "stop"})

    work_dir = tmp_path / "seed-work"
    work_dir.mkdir()
    with pytest.raises(StudioJobRejected, match="escapes"):
        manager._write_seed_inputs(work_dir, {"seed.bin": b"x"}, seed_dir="../escape")
    with pytest.raises(StudioJobRejected, match="must be bytes"):
        manager._write_seed_inputs(work_dir, {"seed.bin": cast(bytes, object())})


def test_studio_job_path_and_environment_helpers_cover_platform_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handle cross-drive common paths and an already complete worker path."""

    assert jobs_module._is_confined_path(root="/absolute", child="relative") is False
    src_path = Path(jobs_module.__file__).resolve().parents[3]
    expected_pythonpath = os.pathsep.join((str(src_path), str(src_path.parent)))
    monkeypatch.setenv("PYTHONPATH", expected_pythonpath)
    environment = jobs_module._process_worker_environment()
    assert environment["PYTHONPATH"] == expected_pythonpath
