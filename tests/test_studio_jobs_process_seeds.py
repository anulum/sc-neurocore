# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio process job seed inputs

"""Seed inputs reach the worker confined, bounded and exactly as submitted."""

from __future__ import annotations


import threading
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")


from sc_neurocore.studio.platform.jobs import (
    STUDIO_SEED_INPUT_DIR,
    StudioJobArtifactUnavailable,
    StudioJobContext,
    StudioJobManager,
    StudioJobRejected,
)


def test_submit_process_task_delivers_confined_seed_inputs(tmp_path: Path) -> None:
    """Process workers read confined seed inputs written at submission time."""
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=15.0,
    )

    record = manager.submit_process_task(
        kind="training",
        owner="studio-training-attach",
        request_id="req-seed",
        task_path="tests.studio_job_tasks:process_seed_echo_task",
        payload={"seed_path": "weights/model.bin"},
        seed_inputs={"weights/model.bin": b"seed payload bytes"},
    )
    completed = manager.wait(record.job_id, timeout_seconds=20.0)

    assert completed.status == "completed"
    assert completed.result == {"seed_text": "seed payload bytes", "seed_bytes": 18}
    # Seed inputs are job inputs, not outputs, so they are never published.
    assert completed.artifacts == ()
    assert str(tmp_path) not in str(completed.to_public_dict())


def test_submit_process_task_rejects_oversized_seed_input(tmp_path: Path) -> None:
    """The advertised seed boundary admits exact bytes and refuses one byte more."""
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=5.0,
        max_artifact_bytes=8,
        max_concurrent_jobs=1,
        max_queued_jobs=0,
    )
    seed = b"s" * manager.max_seed_file_bytes

    try:
        with pytest.raises(StudioJobRejected, match="seed input exceeds"):
            manager.submit_process_task(
                kind="training",
                owner="studio-training-attach",
                request_id="req-big",
                task_path="tests.studio_job_tasks:process_seed_echo_task",
                payload={"seed_path": "weights/model.bin"},
                seed_inputs={"weights/model.bin": seed + b"s"},
            )
        refused = manager.list_records()
        assert len(refused) == 1 and refused[0].status == "failed"
        assert manager.status().active_count == 0
        assert manager.unreaped_workers == ()

        record = manager.submit_process_task(
            kind="training",
            owner="studio-training-attach",
            request_id="req-boundary",
            task_path="tests.studio_job_tasks:process_seed_echo_task",
            payload={"seed_path": "weights/model.bin"},
            seed_inputs={"weights/model.bin": seed},
        )
        completed = manager.wait(record.job_id, timeout_seconds=20.0)
        assert completed.status == "completed", completed.public_error
        assert completed.result == {"seed_text": seed.decode("utf-8"), "seed_bytes": len(seed)}
        assert completed.artifacts == ()
        assert manager.status().active_count == 0
        assert manager.unreaped_workers == ()
    finally:
        for record in manager.list_records():
            manager.cancel(record.job_id)
            manager.wait(record.job_id, timeout_seconds=20.0)
        manager._ledger.close()


def test_submit_process_task_rejects_escaping_seed_path(tmp_path: Path) -> None:
    """Seed inputs whose path escapes the seed directory are rejected."""
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=5.0,
    )

    with pytest.raises(StudioJobRejected, match="escapes the seed directory"):
        manager.submit_process_task(
            kind="training",
            owner="studio-training-attach",
            request_id="req-escape",
            task_path="tests.studio_job_tasks:process_seed_echo_task",
            payload={"seed_path": "model.bin"},
            seed_inputs={"../escape.bin": b"escape"},
        )


def test_read_seed_input_reports_missing_payload(tmp_path: Path) -> None:
    """Reading an absent seed input fails closed."""
    context = StudioJobContext(
        job_id="sj_seed",
        work_dir=tmp_path / "seed-job",
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )

    with pytest.raises(StudioJobArtifactUnavailable, match="seed input is unavailable"):
        context.read_seed_input("model.bin")


def test_read_seed_input_rejects_escaping_path(tmp_path: Path) -> None:
    """Reading a seed input whose path escapes the seed directory fails closed."""
    work_dir = tmp_path / "seed-job"
    (work_dir / STUDIO_SEED_INPUT_DIR).mkdir(parents=True)
    context = StudioJobContext(
        job_id="sj_seed",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )

    with pytest.raises(ValueError, match="escapes the seed directory"):
        context.read_seed_input("../escape.bin")


def test_read_seed_input_rejects_symlinked_seed_directory(tmp_path: Path) -> None:
    """Seed reads reject reserved seed directories that resolve outside the job."""
    work_dir = tmp_path / "seed-job"
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    work_dir.mkdir()
    (work_dir / STUDIO_SEED_INPUT_DIR).symlink_to(outside_dir, target_is_directory=True)
    (outside_dir / "model.bin").write_bytes(b"escape")
    context = StudioJobContext(
        job_id="sj_seed",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )

    with pytest.raises(ValueError, match="escapes the seed directory"):
        context.read_seed_input("model.bin")
