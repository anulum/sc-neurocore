# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job admission control

"""How many jobs run at once, and what the overflow is told.

The failure these guard against is not a slow Studio: it is a Studio that
accepts everything, starts everything, and then cannot answer anything. A
refusal with a reason is a better answer than an admission that never
completes.
"""

from __future__ import annotations

import threading
import subprocess
import sys
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobContext, StudioJobManager
from sc_neurocore.studio.platform.jobs_admission import (
    StudioJobQueueFull,
)


class TestSharedAdmission:
    def _manager(self, root: Path, **kwargs: object) -> StudioJobManager:
        return StudioJobManager(
            root=root,
            allowed_kinds=frozenset({"analysis"}),
            default_timeout_seconds=30.0,
            **kwargs,  # type: ignore[arg-type]
        )

    def test_simultaneous_duplicate_admission_uses_one_slot(self, tmp_path: Path) -> None:
        """Two competing managers atomically converge on one job and reservation."""
        managers = [
            self._manager(tmp_path / "jobs", max_concurrent_jobs=1, max_queued_jobs=0)
            for _ in range(2)
        ]
        barrier = threading.Barrier(2)
        release = threading.Event()
        ids: list[str] = []
        errors: list[BaseException] = []
        executions: list[str] = []

        def task(context: StudioJobContext) -> dict[str, object]:
            executions.append(context.job_id)
            release.wait(5.0)
            return {}

        def submit(manager: StudioJobManager) -> None:
            try:
                barrier.wait(timeout=2.0)
                record = manager.submit(
                    kind="analysis",
                    owner="operator",
                    request_id=None,
                    task=task,
                    idempotency_key="same-request",
                )
                ids.append(record.job_id)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=submit, args=(manager,)) for manager in managers]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=3.0)
                assert not thread.is_alive()
            assert errors == []
            assert len(ids) == 2 and ids[0] == ids[1]
            assert managers[0]._admission.snapshot().running == 1
        finally:
            release.set()
            for thread in threads:
                thread.join(timeout=3.0)
            for job_id in set(ids):
                assert managers[0].wait(job_id, 2.0).status == "completed"
        assert executions == [ids[0]]

    def test_duplicate_live_request_needs_no_new_capacity(self, tmp_path: Path) -> None:
        """An idempotent retry returns the live job even when its slot fills the root."""
        manager = self._manager(tmp_path / "jobs", max_concurrent_jobs=1, max_queued_jobs=0)
        release = threading.Event()

        def blocking(context: StudioJobContext) -> dict[str, object]:
            release.wait(3.0)
            return {}

        first = manager.submit(
            kind="analysis",
            owner="operator",
            request_id=None,
            task=blocking,
            idempotency_key="one-live-job",
        )
        try:
            duplicate = manager.submit(
                kind="analysis",
                owner="operator",
                request_id=None,
                task=blocking,
                idempotency_key="one-live-job",
            )
            assert duplicate.job_id == first.job_id
            assert len(manager.list_records()) == 1
        finally:
            release.set()
            assert manager.wait(first.job_id, 2.0).status == "completed"

    def test_shared_root_limit_applies_in_another_process(self, tmp_path: Path) -> None:
        """A new interpreter sees the occupied capacity, not an empty private counter."""
        manager = self._manager(tmp_path / "jobs", max_concurrent_jobs=1, max_queued_jobs=0)
        started, release = threading.Event(), threading.Event()

        def blocking(context: StudioJobContext) -> dict[str, object]:
            started.set()
            release.wait(10.0)
            return {}

        record = manager.submit(kind="analysis", owner="operator", request_id=None, task=blocking)
        try:
            assert started.wait(1.0)
            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    """
import sys
from pathlib import Path
from sc_neurocore.studio.platform.jobs import StudioJobManager
from sc_neurocore.studio.platform.jobs_admission import StudioJobQueueFull
manager = StudioJobManager(root=Path(sys.argv[1]), allowed_kinds=frozenset({'analysis'}),
    default_timeout_seconds=3.0, max_concurrent_jobs=1, max_queued_jobs=0)
try:
    job = manager.submit(kind='analysis', owner='operator', request_id=None, task=lambda context: {})
except StudioJobQueueFull:
    print('refused')
else:
    assert manager.wait(job.job_id, 2.0).status == 'completed'
    print('accepted')
""",
                    str(manager.root),
                ],
                capture_output=True,
                text=True,
                timeout=5.0,
                check=True,
            )
            assert child.stdout.strip() == "refused", child.stderr
            assert len(manager.list_records()) == 1
        finally:
            release.set()
            assert manager.wait(record.job_id, 2.0).status == "completed"

    def test_shared_root_enforces_one_limit_across_managers(self, tmp_path: Path) -> None:
        """A second manager cannot multiply the same job root's compute budget."""
        first = self._manager(tmp_path / "jobs", max_concurrent_jobs=1, max_queued_jobs=0)
        second = self._manager(tmp_path / "jobs", max_concurrent_jobs=1, max_queued_jobs=0)
        started, release = threading.Event(), threading.Event()
        admitted: list[tuple[StudioJobManager, str]] = []

        def blocking(context: StudioJobContext) -> dict[str, object]:
            started.set()
            release.wait(3.0)
            return {"done": True}

        try:
            record = first.submit(kind="analysis", owner="operator", request_id=None, task=blocking)
            admitted.append((first, record.job_id))
            assert started.wait(1.0)
            with pytest.raises(StudioJobQueueFull):
                unexpected = second.submit(
                    kind="analysis", owner="operator", request_id=None, task=blocking
                )
                admitted.append((second, unexpected.job_id))
            assert len(second.list_records()) == 1
        finally:
            release.set()
            for manager, job_id in admitted:
                assert manager.wait(job_id, 2.0).status == "completed"
