# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared job queue lifecycle

"""Exercise real shared-ledger queues through managers and their admission owner."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobContext, StudioJobManager
from sc_neurocore.studio.platform.jobs_admission import StudioJobQueueFull


def _manager(root: Path) -> StudioJobManager:
    return StudioJobManager(
        root=root,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=15.0,
        max_concurrent_jobs=1,
        max_queued_jobs=2,
    )


def _wait_queued(manager: StudioJobManager, expected: int) -> None:
    deadline = time.monotonic() + 3.0
    while manager._admission.snapshot().queued != expected and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager._admission.snapshot().queued == expected


@pytest.mark.parametrize("duplicate", [False, True])
def test_shared_queue_fifo_and_queued_duplicate(tmp_path: Path, duplicate: bool) -> None:
    """Later submissions cannot overtake the oldest queued job or duplicate its work."""
    managers = [_manager(tmp_path) for _ in range(3)]
    releases = [threading.Event() for _ in managers]
    starts = [threading.Event() for _ in managers]
    executions: list[int] = []
    ids: dict[int, str] = {}
    errors: list[BaseException] = []

    def submit(index: int) -> None:
        def task(context: StudioJobContext) -> dict[str, object]:
            executions.append(index)
            starts[index].set()
            releases[index].wait(10.0)
            return {"index": index}

        try:
            record = managers[index].submit(
                kind="analysis",
                owner="owner",
                request_id=None,
                task=task,
                idempotency_key="same" if duplicate and index > 0 else str(index),
            )
            ids[index] = record.job_id
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=submit, args=(index,)) for index in range(3)]
    try:
        threads[0].start()
        assert starts[0].wait(2.0)
        threads[1].start()
        _wait_queued(managers[0], 1)
        threads[2].start()
        _wait_queued(managers[0], 2)
        with pytest.raises(StudioJobQueueFull):
            managers[0].submit(kind="analysis", owner="extra", request_id=None, task=lambda ctx: {})
        releases[0].set()
        assert starts[1].wait(3.0)
        assert not starts[2].is_set()
        if duplicate:
            threads[2].join(timeout=3.0)
            assert not threads[2].is_alive()
            threads[1].join(timeout=3.0)
            assert ids[2] == ids[1]
            _wait_queued(managers[0], 0)
        else:
            _wait_queued(managers[0], 1)
            releases[1].set()
            assert starts[2].wait(3.0)
    finally:
        for release in releases:
            release.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=12.0)
                assert not thread.is_alive()
        for job_id in set(ids.values()):
            assert managers[0].wait(job_id, 2.0).status == "completed"
    assert not errors
    assert executions == ([0, 1] if duplicate else [0, 1, 2])
    snapshot = managers[0]._admission.snapshot()
    assert snapshot.running == snapshot.queued == 0
    assert snapshot.admitted == len(executions)
    assert snapshot.refused == 1


def test_queued_timeout_removes_only_its_reservation(tmp_path: Path) -> None:
    """A timed-out wait is counted once and cannot release the running owner's job."""
    manager, observer = _manager(tmp_path), _manager(tmp_path)
    release = threading.Event()

    def task(context: StudioJobContext) -> dict[str, object]:
        release.wait(5.0)
        return {}

    record = manager.submit(kind="analysis", owner="owner", request_id=None, task=task)
    try:
        observer._admission.release(job_id=record.job_id)
        assert manager._admission.snapshot().running == 1
        with pytest.raises(StudioJobQueueFull):
            observer._admission.admit(
                job_id="sj_0000000000000001",
                kind="analysis",
                actor="observer",
                workspace="default",
                request_id=None,
                idempotency_key=None,
                experiment_sha256=None,
                admission=None,
                execution_model="thread",
                timeout_seconds=0.1,
            )
        observer._admission.release(job_id="sj_0000000000000001")
        snapshot = manager._admission.snapshot()
        assert snapshot.running == 1 and snapshot.queued == 0
        assert snapshot.admitted == snapshot.refused == 1
        assert [item.job_id for item in manager.list_records()] == [record.job_id]
    finally:
        release.set()
        assert manager.wait(record.job_id, 2.0).status == "completed"
