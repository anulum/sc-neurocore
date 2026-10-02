# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Live ledger observation recovery

"""Exercise real row loss and repair while public job supervisors are running."""

import json
import time
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobManager
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from tests.studio_jobs_failure_tasks import restore_record_on_cancellation


def _await_file(path: Path) -> None:
    """Wait for an actual worker checkpoint with a bounded deadline."""
    deadline = time.monotonic() + 20.0
    while not path.is_file():
        assert time.monotonic() < deadline, f"Worker did not write {path.name}"
        time.sleep(0.01)


@pytest.mark.parametrize("mode", ["thread", "process"])
def test_missing_live_record_stops_worker_and_retains_safe_failure(
    tmp_path: Path, mode: str
) -> None:
    """An actual observation error stops work, then seals the repaired original row."""
    manager = StudioJobManager(
        root=tmp_path,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=30.0,
        max_concurrent_jobs=1,
        max_queued_jobs=0,
    )
    payload: dict[str, object] = {"root": str(tmp_path), "mode": mode}
    if mode == "thread":
        submitted = manager.submit(
            kind="analysis",
            owner="operator",
            request_id=None,
            task=lambda context: restore_record_on_cancellation(context, payload),
        )
    else:
        submitted = manager.submit_process_task(
            kind="analysis",
            owner="operator",
            request_id=None,
            task_path="tests.studio_jobs_failure_tasks:restore_record_on_cancellation",
            payload=payload,
        )
    work_dir = tmp_path / submitted.job_id
    ledger = StudioJobLedger(root=tmp_path)
    original: dict[str, object] = {}
    try:
        _await_file(work_dir / "ready.txt")
        history = manager.transitions(submitted.job_id)
        with ledger.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id=?", (submitted.job_id,)
            ).fetchone()
            assert row is not None
            original = dict(row)
            (work_dir / ".restore_row.json").write_text(json.dumps(original))
            connection.execute("DELETE FROM jobs WHERE job_id=?", (submitted.job_id,))
        _await_file(work_dir / "restored.txt")
        terminal = manager.wait(submitted.job_id, 20.0)
        assert terminal.status == "failed" and terminal.error is not None
        assert submitted.job_id in terminal.error
        public = terminal.to_public_dict()["error"]
        suffix = "Worker stopped: True." if mode == "thread" else "Worker reaped."
        assert public == "Studio cancellation observation failed. " + suffix
        assert str(tmp_path) not in str(public) and submitted.job_id not in str(public)
        transitions = manager.transitions(submitted.job_id)
        assert transitions[: len(history)] == history and len(transitions) == len(history) + 1
        assert manager.status().active_count == 0 and manager.unreaped_workers == ()
        assert (work_dir / "ready.txt").read_text() == "worker ready"
        assert (work_dir / "restored.txt").read_text() == "original row restored"
        assert {artifact.relative_path for artifact in terminal.artifacts} == {
            "ready.txt",
            "restored.txt",
        }
    finally:
        if original:
            columns = ", ".join(original)
            placeholders = ", ".join("?" for _ in original)
            with ledger.transaction() as connection:
                connection.execute(
                    f"INSERT OR IGNORE INTO jobs ({columns}) VALUES ({placeholders})",
                    tuple(original.values()),
                )
        manager.cancel(submitted.job_id)
        manager.wait(submitted.job_id, 20.0)
        ledger.close()
