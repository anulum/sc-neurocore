# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job custody when the operating system does not cooperate

"""Every job ends in a record that says what happened, even when the host refuses.

The failures here are ones a shared, unprivileged test process cannot produce on
demand: a thread the interpreter will not start, a process group that outlives
its reap, a ``/proc`` entry that disappears between two reads, a rename that
loses a race. Each is injected at the one boundary that fails, with the real
job manager, ledger, worker processes and directories around it, and each case
asserts the record the operator would read.
"""

from __future__ import annotations

import ctypes
import os
import signal
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest

from sc_neurocore.studio.platform import (
    jobs_ledger_supervisor,
    jobs_manager_process,
    jobs_manager_thread,
    jobs_process_protocol,
    jobs_process_state,
    jobs_purge,
    jobs_purge_paths,
    jobs_purge_recovery,
    jobs_reaper,
    jobs_worker_recovery,
    jobs_worker_registration,
)
from sc_neurocore.studio.platform.jobs import StudioJobContext, StudioJobManager
from sc_neurocore.studio.platform.jobs_ledger_supervisor import (
    supervisor_identity,
    supervisor_is_alive,
)
from sc_neurocore.studio.platform.jobs_models import StudioJobRejected
from sc_neurocore.studio.platform.jobs_reaper import ReapReport, reap_process_group

#: A process id no live process holds on this host during the test.
ABSENT_PID = 4_194_000


def _absent_pid() -> int:
    assert not Path(f"/proc/{ABSENT_PID}").exists()
    return ABSENT_PID


class RefusedThread(threading.Thread):
    """A thread the interpreter refuses to start, as it does at its thread limit."""

    def start(self) -> None:
        raise RuntimeError("can't start new thread")


class StartedThenRefusedThread(threading.Thread):
    """A thread that starts and whose start still reports an error."""

    def start(self) -> None:
        super().start()
        raise RuntimeError("thread start reported an error after starting")


def _threading_with(thread_class: type[threading.Thread]) -> types.SimpleNamespace:
    return types.SimpleNamespace(Thread=thread_class, Event=threading.Event)


def _manager(root: Path, **options: Any) -> StudioJobManager:
    return StudioJobManager(
        root=root,
        allowed_kinds=frozenset({"analysis", "compiler"}),
        default_timeout_seconds=options.pop("timeout", 15.0),
        **options,
    )


# ── supervisor liveness ──────────────────────────────────────────────


class TestSupervisorLiveness:
    def _identity(self, pid: int) -> str:
        return f"{socket.gethostname()}:{pid}:12345"

    @pytest.mark.parametrize(
        ("recheck", "expected"),
        [
            (ProcessLookupError(), False),
            (PermissionError(), None),
            (None, None),
        ],
        ids=["gone", "unreadable", "still-signalable"],
    )
    def test_a_process_whose_metadata_vanished_after_the_probe(
        self, monkeypatch: pytest.MonkeyPatch, recheck: BaseException | None, expected: bool | None
    ) -> None:
        """kill(0) answered, then /proc had no entry: the second probe decides."""
        calls: list[int] = []

        def kill(pid: int, number: int) -> None:
            calls.append(pid)
            if len(calls) > 1 and recheck is not None:
                raise recheck

        monkeypatch.setattr(os, "kill", kill)
        assert supervisor_is_alive(self._identity(_absent_pid())) is expected
        assert len(calls) == 2

    def test_unreadable_process_metadata_is_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def refused(*_args: object, **_kwargs: object) -> None:
            raise PermissionError("proc entry of another identity")

        monkeypatch.setattr(jobs_ledger_supervisor, "open", refused, raising=False)
        assert supervisor_is_alive(self._identity(os.getpid())) is None


# ── process groups ───────────────────────────────────────────────────


def test_a_group_this_identity_may_not_signal_is_not_gone(monkeypatch: pytest.MonkeyPatch) -> None:
    def killpg(group: int, number: int) -> None:
        raise PermissionError("operation not permitted")

    monkeypatch.setattr(os, "killpg", killpg)
    assert jobs_process_state.group_is_gone(12345) is False


def test_a_process_that_exits_after_the_listing_is_not_a_survivor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A /proc listing can name a process that exits before it is probed.

    That exit happens between two system calls, so the listing is made stale
    instead: it names a process that no longer exists, and the real
    ``getpgid`` then refuses it exactly as it does after such an exit.
    """
    absent = _absent_pid()
    real_listdir = os.listdir

    def stale_listing(path: str) -> list[str]:
        entries = real_listdir(path)
        return [str(absent), *entries] if path == "/proc" else entries

    monkeypatch.setattr(os, "listdir", stale_listing)
    assert jobs_process_state.group_survivors(absent) == ()


class TestWorkerGroupStopped:
    def _call(self) -> bool:
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        group = _absent_pid()
        return jobs_worker_recovery.worker_group_stopped(
            f"{socket.gethostname()}:{group}:777", boot, group
        )

    @pytest.mark.parametrize(
        ("recheck", "expected"),
        [(ProcessLookupError(), True), (None, False)],
        ids=["vanished", "still-present"],
    )
    def test_a_group_without_visible_members_is_rechecked(
        self, monkeypatch: pytest.MonkeyPatch, recheck: BaseException | None, expected: bool
    ) -> None:
        """Members that leave during enumeration, or belong elsewhere, prove nothing."""
        own = os.getpid()
        probes: list[int] = []

        def killpg(group: int, number: int) -> None:
            probes.append(group)
            if len(probes) > 1 and recheck is not None:
                raise recheck

        def getpgid(pid: int) -> int:
            if pid == own:
                # Claims the probed group, but this process's own stat says otherwise.
                return ABSENT_PID
            raise ProcessLookupError(pid)

        monkeypatch.setattr(os, "killpg", killpg)
        monkeypatch.setattr(os, "getpgid", getpgid)
        assert self._call() is expected
        assert probes == [ABSENT_PID, ABSENT_PID]


# ── reaping ──────────────────────────────────────────────────────────


class UnstoppableChild:
    """A direct child in the caller's own group whose signals are refused."""

    def __init__(self, *, refuse_terminate: bool, refuse_kill: bool) -> None:
        self.pid = os.getpid()
        self.returncode: int | None = None
        self._refuse_terminate = refuse_terminate
        self._refuse_kill = refuse_kill

    def poll(self) -> int | None:
        return None

    def terminate(self) -> None:
        if self._refuse_terminate:
            raise PermissionError("signal refused by a security policy")

    def kill(self) -> None:
        if self._refuse_kill:
            raise PermissionError("signal refused by a security policy")

    def wait(self, timeout: float | None = None) -> int:
        raise subprocess.TimeoutExpired("worker", timeout or 0.0)


class TestReaper:
    def test_signalling_a_group_that_is_gone_is_not_an_error(self) -> None:
        jobs_reaper._signal_group(_absent_pid(), signal.SIGTERM)

    @pytest.mark.parametrize(
        ("refuse_terminate", "refuse_kill"),
        [(True, False), (False, True), (False, False)],
        ids=["terminate-refused", "kill-refused", "both-ignored"],
    )
    def test_a_child_sharing_the_supervisor_group_that_will_not_stop_is_reported(
        self, refuse_terminate: bool, refuse_kill: bool
    ) -> None:
        # A stand-in for the one Popen that cannot be produced: the kernel refusing both signals.
        child: Any = UnstoppableChild(refuse_terminate=refuse_terminate, refuse_kill=refuse_kill)
        report = reap_process_group(
            child,
            terminate_grace_seconds=0.01,
            kill_grace_seconds=0.01,
        )
        assert report.outcome == "unreaped"
        assert report.survivors == (os.getpid(),)
        assert report.group_id is None

    def test_collecting_a_child_that_is_still_running_does_not_block(self) -> None:
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            started = time.monotonic()
            jobs_reaper._collect(child)
            assert child.returncode is None
            assert time.monotonic() - started < 5.0
        finally:
            child.kill()
            child.wait(timeout=10)


# ── thread-backed jobs ───────────────────────────────────────────────


class TestThreadStartRefused:
    def test_a_supervisor_that_cannot_start_fails_the_job(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _manager(tmp_path / "jobs")
        monkeypatch.setattr(jobs_manager_thread, "threading", _threading_with(RefusedThread))
        with pytest.raises(RuntimeError, match="can't start new thread"):
            manager.submit(kind="analysis", owner="owner", request_id=None, task=lambda ctx: {})
        (record,) = manager.list_records()
        assert record.status == "failed"
        assert record.error == "Studio job could not start: can't start new thread"
        assert manager._done_events[record.job_id].is_set()

    def test_a_worker_that_cannot_start_fails_the_job(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _manager(tmp_path / "jobs")
        created: list[int] = []

        class SecondRefused(threading.Thread):
            def start(self) -> None:
                created.append(1)
                if len(created) > 1:
                    raise RuntimeError("can't start new thread")
                super().start()

        monkeypatch.setattr(jobs_manager_thread, "threading", _threading_with(SecondRefused))
        job = manager.submit(kind="analysis", owner="owner", request_id=None, task=lambda ctx: {})
        record = manager.wait(job.job_id, timeout_seconds=10.0)
        assert record.status == "failed"
        assert record.error == "Studio worker could not start: can't start new thread"

    def test_a_worker_whose_start_errs_after_starting_is_still_supervised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _manager(tmp_path / "jobs")
        created: list[int] = []

        class SecondErrs(threading.Thread):
            def start(self) -> None:
                created.append(1)
                super().start()
                if len(created) > 1:
                    raise RuntimeError("thread start reported an error after starting")

        def task(context: StudioJobContext) -> dict[str, object]:
            deadline = time.monotonic() + 5.0
            while not context.cancelled and time.monotonic() < deadline:
                time.sleep(0.01)
            return {"stopped_by_cancellation": context.cancelled}

        monkeypatch.setattr(jobs_manager_thread, "threading", _threading_with(SecondErrs))
        job = manager.submit(kind="analysis", owner="owner", request_id=None, task=task)
        record = manager.wait(job.job_id, timeout_seconds=10.0)
        # The worker ran, was told to stop, and the start error is what the record keeps.
        assert record.status == "failed"
        assert record.error == "thread start reported an error after starting"


# ── process-backed jobs ──────────────────────────────────────────────


def _unreaped_after_reaping(
    monkeypatch: pytest.MonkeyPatch,
) -> list[ReapReport]:
    """Reap for real, then report the group as surviving, as a stuck descendant would."""
    reports: list[ReapReport] = []

    def reap(process: subprocess.Popen[bytes], **options: Any) -> ReapReport:
        real = reap_process_group(process, **options)
        report = ReapReport(
            outcome="unreaped",
            group_id=real.group_id,
            returncode=real.returncode,
            duration_seconds=real.duration_seconds,
            survivors=(process.pid,),
        )
        reports.append(report)
        return report

    monkeypatch.setattr(jobs_process_protocol, "reap_process_group", reap)
    return reports


def _submit(
    manager: StudioJobManager, task: str, payload: dict[str, object], **options: Any
) -> str:
    record = manager.submit_process_task(
        kind="compiler",
        owner="operator",
        request_id=None,
        task_path=f"tests.studio_job_tasks:{task}",
        payload=payload,
        **options,
    )
    return record.job_id


class TestProcessSupervision:
    def test_a_worker_whose_registration_cannot_start_is_reaped_and_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _manager(tmp_path / "jobs")
        reports = _unreaped_after_reaping(monkeypatch)

        def refuse(*_args: object) -> None:
            raise RuntimeError("registration thread refused")

        monkeypatch.setattr(jobs_process_protocol, "start_worker_registration", refuse)
        job_id = _submit(manager, "process_sleep_task", {"seconds": 10.0})
        record = manager.wait(job_id, timeout_seconds=30.0)
        assert record.status == "failed"
        assert record.error is not None
        assert record.error.startswith("Studio worker could not start: registration thread refused")
        assert "was not reaped" in record.error
        assert len(reports) == 1

    def test_a_ledger_that_cannot_be_read_fails_the_job_and_stops_the_worker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _manager(tmp_path / "jobs")
        reports = _unreaped_after_reaping(monkeypatch)
        job_id = _submit(manager, "process_sleep_task", {"seconds": 10.0})
        deadline = time.monotonic() + 30.0
        while manager.record(job_id).status != "running":
            assert time.monotonic() < deadline
            time.sleep(0.02)
        real_record = manager._ledger.record

        def unreadable(requested: str) -> Any:
            if requested == job_id and threading.current_thread() is not threading.main_thread():
                raise sqlite3.OperationalError("disk I/O error")
            return real_record(requested)

        monkeypatch.setattr(manager._ledger, "record", unreadable)
        assert manager._done_events[job_id].wait(30.0)
        monkeypatch.setattr(manager._ledger, "record", real_record)
        record = manager.record(job_id)
        assert record.status == "failed"
        assert record.error is not None
        assert record.error.startswith("Studio cancellation observation failed: disk I/O error.")
        assert "was not reaped" in record.error
        assert reports

    def test_a_cancelled_worker_that_survives_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _manager(tmp_path / "jobs")
        _unreaped_after_reaping(monkeypatch)
        job_id = _submit(manager, "process_sleep_task", {"seconds": 10.0})
        deadline = time.monotonic() + 30.0
        while manager.record(job_id).status != "running":
            assert time.monotonic() < deadline
            time.sleep(0.02)
        manager.cancel(job_id)
        record = manager.wait(job_id, timeout_seconds=30.0)
        assert record.status == "cancelled"
        assert record.error is not None and "was not reaped" in record.error

    def test_a_timed_out_worker_that_survives_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _manager(tmp_path / "jobs")
        _unreaped_after_reaping(monkeypatch)
        job_id = _submit(manager, "process_sleep_task", {"seconds": 10.0}, timeout_seconds=1.0)
        record = manager.wait(job_id, timeout_seconds=30.0)
        assert record.status == "timed_out"
        assert record.error is not None
        assert record.error.startswith("Studio job exceeded its timeout. The worker process group")

    def test_a_finished_worker_whose_group_survives_is_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _manager(tmp_path / "jobs")
        _unreaped_after_reaping(monkeypatch)
        job_id = _submit(manager, "process_echo_task", {"model": "lif"})
        record = manager.wait(job_id, timeout_seconds=30.0)
        assert record.status == "failed"
        assert record.error is not None and record.error.startswith(
            "The worker process group was not reaped"
        )

    def test_a_process_supervisor_that_cannot_start_fails_the_job(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _manager(tmp_path / "jobs")
        monkeypatch.setattr(jobs_manager_process, "threading", _threading_with(RefusedThread))
        with pytest.raises(RuntimeError, match="can't start new thread"):
            _submit(manager, "process_echo_task", {"model": "lif"})
        (record,) = manager.list_records()
        assert record.status == "failed"
        assert manager._done_events[record.job_id].is_set()


def test_a_registration_thread_that_cannot_start_closes_the_worker_pipe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path / "jobs")
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE
    )
    try:
        monkeypatch.setattr(jobs_worker_registration, "threading", _threading_with(RefusedThread))
        with pytest.raises(RuntimeError, match="can't start new thread"):
            jobs_worker_registration.start_worker_registration(
                manager._ledger, "sj_0000000000000000", child, supervisor_identity()
            )
        assert child.stdin is not None and child.stdin.closed
        # The worker sees end of input instead of waiting for a grant forever.
        assert child.wait(timeout=10) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)


# ── purge ────────────────────────────────────────────────────────────


def _completed_job(manager: StudioJobManager) -> str:
    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("proof.txt", "retained evidence")
        return {}

    job = manager.submit(kind="analysis", owner="owner", request_id=None, task=task)
    assert manager.wait(job.job_id, 5.0).status == "completed"
    assert manager._done_events[job.job_id].wait(5.0)
    return job.job_id


class TestOwnedDirectoryRemoval:
    def test_a_missing_directory_is_not_reported_removed(self, tmp_path: Path) -> None:
        assert jobs_purge_recovery._remove_owned_directory(tmp_path / "missing", 1, 1) is False

    def test_a_directory_with_another_identity_is_left_alone(self, tmp_path: Path) -> None:
        target = tmp_path / "staged"
        target.mkdir()
        (target / "kept.txt").write_text("not ours")
        identity = target.stat()
        assert (
            jobs_purge_recovery._remove_owned_directory(
                target, identity.st_dev, identity.st_ino + 1
            )
            is False
        )
        assert (target / "kept.txt").read_text() == "not ours"

    def test_a_path_exchanged_while_it_was_cleared_is_not_removed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        target = tmp_path / "staged"
        target.mkdir()
        identity = target.stat()
        monkeypatch.setattr(jobs_purge_recovery, "_matches", lambda *_args: False)
        assert (
            jobs_purge_recovery._remove_owned_directory(target, identity.st_dev, identity.st_ino)
            is False
        )
        assert target.is_dir()


class TestPurgeRecoveryPhases:
    def _departed(self) -> str:
        return subprocess.run(
            [
                sys.executable,
                "-c",
                "from sc_neurocore.studio.platform.jobs_ledger_supervisor "
                "import supervisor_identity; print(supervisor_identity())",
            ],
            capture_output=True,
            text=True,
            timeout=30.0,
            check=True,
        ).stdout.strip()

    def _journal(self, manager: StudioJobManager, job_id: str, state: str, identity: Any) -> None:
        with manager._ledger.transaction() as connection:
            connection.execute(
                "INSERT INTO job_purges VALUES(?,?,?,?,?)",
                (job_id, self._departed(), identity.st_dev, identity.st_ino, state),
            )

    def _pending(self, manager: StudioJobManager, job_id: str) -> str | None:
        row = (
            manager._ledger.connection()
            .execute("SELECT state FROM job_purges WHERE job_id=?", (job_id,))
            .fetchone()
        )
        return None if row is None else str(row["state"])

    def test_a_staged_directory_is_restored_when_the_purge_never_committed(
        self, tmp_path: Path
    ) -> None:
        manager = _manager(tmp_path)
        job_id = _completed_job(manager)
        work_dir = tmp_path / job_id
        identity = work_dir.stat()
        work_dir.rename(tmp_path / f".purge-{job_id}")
        self._journal(manager, job_id, "prepared", identity)
        assert jobs_purge_recovery.recover_purges(manager) == (job_id,)
        assert (work_dir / "proof.txt").read_text() == "retained evidence"
        assert self._pending(manager, job_id) is None

    def test_both_the_original_and_a_staged_copy_leave_the_intent_pending(
        self, tmp_path: Path
    ) -> None:
        manager = _manager(tmp_path)
        job_id = _completed_job(manager)
        identity = (tmp_path / job_id).stat()
        (tmp_path / f".purge-{job_id}").mkdir()
        self._journal(manager, job_id, "prepared", identity)
        assert jobs_purge_recovery.recover_purges(manager) == ()
        assert self._pending(manager, job_id) == "prepared"

    def test_a_restore_that_loses_its_race_leaves_the_intent_pending(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _manager(tmp_path)
        job_id = _completed_job(manager)
        work_dir = tmp_path / job_id
        identity = work_dir.stat()
        work_dir.rename(tmp_path / f".purge-{job_id}")
        self._journal(manager, job_id, "prepared", identity)
        monkeypatch.setattr(jobs_purge_paths, "move_without_replace", lambda *_args: False)
        assert jobs_purge_recovery.recover_purges(manager) == ()
        assert self._pending(manager, job_id) == "prepared"

    def test_a_cleanup_that_cannot_prove_removal_is_marked_ambiguous(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _manager(tmp_path)
        job_id = _completed_job(manager)
        work_dir = tmp_path / job_id
        identity = work_dir.stat()
        work_dir.rename(tmp_path / f".purge-{job_id}")
        manager._ledger.delete(job_id)
        self._journal(manager, job_id, "cleanup_started", identity)
        monkeypatch.setattr(jobs_purge_recovery, "_remove_owned_directory", lambda *_args: False)
        assert jobs_purge_recovery.recover_purges(manager) == ()
        assert self._pending(manager, job_id) == "ambiguous"

    def test_recovery_for_one_job_leaves_other_intents_untouched(self, tmp_path: Path) -> None:
        manager = _manager(tmp_path)
        first, second = _completed_job(manager), _completed_job(manager)
        for job_id in (first, second):
            identity = (tmp_path / job_id).stat()
            (tmp_path / job_id).rename(tmp_path / f".purge-{job_id}")
            self._journal(manager, job_id, "prepared", identity)
        assert jobs_purge_recovery.recover_purges(manager, own_job=first) == (first,)
        assert self._pending(manager, second) == "prepared"


class TestPurgeCommit:
    def test_a_move_that_loses_its_race_rejects_the_purge_and_keeps_the_job(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _manager(tmp_path)
        job_id = _completed_job(manager)
        monkeypatch.setattr(jobs_purge_paths, "move_without_replace", lambda *_args: False)
        with pytest.raises(StudioJobRejected, match="pending purge requiring recovery"):
            manager.purge_terminal_record(job_id)
        assert manager.record(job_id).status == "completed"
        assert (tmp_path / job_id / "proof.txt").read_text() == "retained evidence"

    def test_a_directory_exchanged_during_the_move_rejects_the_purge(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _manager(tmp_path)
        job_id = _completed_job(manager)
        monkeypatch.setattr(jobs_purge, "_matches", lambda *_args: False)
        with pytest.raises(StudioJobRejected, match="identity changed during move"):
            manager.purge_terminal_record(job_id)
        assert manager.record(job_id).status == "completed"

    def test_a_job_deleted_between_preparation_and_commit_is_forgotten(
        self, tmp_path: Path
    ) -> None:
        manager = _manager(tmp_path)
        job_id = _completed_job(manager)
        statements: list[str] = []
        other = sqlite3.connect(manager._ledger.path, timeout=10.0)

        def delete_after_prepare(statement: str) -> None:
            if (
                any("'prepared'" in text for text in statements)
                and statements[-1] == "COMMIT"
                and statement.startswith("BEGIN")
            ):
                other.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
                other.commit()
            statements.append(statement)

        connection = manager._ledger.connection()
        connection.set_trace_callback(delete_after_prepare)
        try:
            # The record another writer removed is reported missing, as for any unknown job.
            with pytest.raises(KeyError, match=job_id):
                manager.purge_terminal_record(job_id)
        finally:
            connection.set_trace_callback(None)
            other.close()
        assert job_id not in manager._done_events

    def test_a_cleanup_left_pending_is_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _manager(tmp_path)
        job_id = _completed_job(manager)
        monkeypatch.setattr(jobs_purge_recovery, "_remove_owned_directory", lambda *_args: False)
        with pytest.raises(StudioJobRejected, match="cleanup remains pending recovery"):
            manager.purge_terminal_record(job_id)


# ── remaining boundaries ─────────────────────────────────────────────


def test_a_transaction_that_cannot_begin_rolls_nothing_back(tmp_path: Path) -> None:
    """Another writer holds the database; BEGIN fails and no transaction exists to undo."""
    manager = _manager(tmp_path)
    ledger = manager._ledger
    ledger.connection().execute("PRAGMA busy_timeout=0")
    holder = sqlite3.connect(ledger.path, timeout=0.0, isolation_level=None)
    holder.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"), ledger.transaction():
            pass
        assert not ledger.connection().in_transaction
    finally:
        holder.execute("ROLLBACK")
        holder.close()


class TestProcessStart:
    def test_a_worker_the_host_cannot_fork_fails_the_job(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _manager(tmp_path / "jobs")

        def refused(*_args: object, **_kwargs: object) -> None:
            raise OSError(11, "Resource temporarily unavailable")

        monkeypatch.setattr(subprocess, "Popen", refused)
        job_id = _submit(manager, "process_echo_task", {"model": "lif"})
        record = manager.wait(job_id, timeout_seconds=30.0)
        assert record.status == "failed"
        assert record.error == (
            "Studio worker could not start: [Errno 11] Resource temporarily unavailable"
        )

    def test_a_refused_registration_whose_worker_is_reaped_says_only_why(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _manager(tmp_path / "jobs")

        def refuse(*_args: object) -> None:
            raise RuntimeError("registration thread refused")

        monkeypatch.setattr(jobs_process_protocol, "start_worker_registration", refuse)
        job_id = _submit(manager, "process_sleep_task", {"seconds": 10.0})
        record = manager.wait(job_id, timeout_seconds=30.0)
        assert record.status == "failed"
        assert record.error == "Studio worker could not start: registration thread refused"


def test_a_thread_whose_status_cannot_be_read_counts_as_exited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unreadable(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("task stat of another identity")

    monkeypatch.setattr(jobs_process_state, "open", unreadable, raising=False)
    assert jobs_process_state.process_exited(os.getpid()) is True


class TestNonReplacingMove:
    def test_an_occupied_destination_is_refused_not_overwritten(self, tmp_path: Path) -> None:
        source, destination = tmp_path / "source", tmp_path / "destination"
        source.mkdir()
        destination.mkdir()
        (destination / "kept.txt").write_text("existing")
        assert jobs_purge_paths.move_without_replace(source, destination) is False
        assert source.is_dir() and (destination / "kept.txt").read_text() == "existing"

    def test_a_missing_source_raises_with_both_paths(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError) as raised:
            jobs_purge_paths.move_without_replace(tmp_path / "missing", tmp_path / "target")
        assert raised.value.filename == str(tmp_path / "missing")
        assert raised.value.filename2 == str(tmp_path / "target")

    def test_a_libc_without_renameat2_refuses_rather_than_falling_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(ctypes, "CDLL", lambda *_args, **_kwargs: types.SimpleNamespace())
        with pytest.raises(OSError, match="Atomic non-overwriting rename is unavailable"):
            jobs_purge_paths.move_without_replace(tmp_path / "a", tmp_path / "b")


def test_a_restored_directory_with_another_identity_leaves_the_intent_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phases = TestPurgeRecoveryPhases()
    manager = _manager(tmp_path)
    job_id = _completed_job(manager)
    work_dir = tmp_path / job_id
    identity = work_dir.stat()
    work_dir.rename(tmp_path / f".purge-{job_id}")
    phases._journal(manager, job_id, "prepared", identity)
    real = jobs_purge_recovery._matches

    def exchanged_after_move(path: Path, device: int | None, inode: int | None) -> bool:
        # The staged directory is ours; what arrives at the original path is not.
        return real(path, device, inode) if path.name.startswith(".purge-") else False

    monkeypatch.setattr(jobs_purge_recovery, "_matches", exchanged_after_move)
    assert jobs_purge_recovery.recover_purges(manager) == ()
    assert phases._pending(manager, job_id) == "prepared"


def test_a_group_that_outlives_sigkill_is_reported_unreaped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A member stuck in the kernel keeps its group; SIGKILL cannot make it leave."""
    worker = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        monkeypatch.setattr(jobs_reaper, "group_is_gone", lambda _group: False)
        report = reap_process_group(
            worker,
            owned_group_id=worker.pid,
            terminate_grace_seconds=0.05,
            kill_grace_seconds=0.05,
        )
        assert report.outcome == "unreaped"
        assert report.group_id == worker.pid
    finally:
        if worker.poll() is None:
            worker.kill()
        worker.wait(timeout=10)
