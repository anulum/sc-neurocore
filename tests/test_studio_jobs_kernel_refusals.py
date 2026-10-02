# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual worker resource refusals

"""Exercise worker startup after a real supervisor already owns the job."""

from pathlib import Path
from textwrap import dedent

from tests.studio_seccomp_support import run_child


def test_worker_thread_limit_keeps_private_cause_and_releases_capacity(tmp_path: Path) -> None:
    """Lower only a child's soft process limit after its supervisor starts."""
    result = run_child(
        dedent(
            """
            import resource, sys, threading
            from pathlib import Path
            from sc_neurocore.studio.platform.jobs import StudioJobManager
            from tests.studio_syscall_support import finish, hold_system_calls

            manager = StudioJobManager(root=Path(sys.argv[1]),
                allowed_kinds=frozenset({'analysis'}), default_timeout_seconds=15.0,
                max_concurrent_jobs=1, max_queued_jobs=0)
            original = resource.getrlimit(resource.RLIMIT_NPROC)
            main_thread = threading.get_native_id()
            observation = {'lowered': False, 'callback_error': None}
            def decide(call):
                if (not observation['lowered'] and call.thread != main_thread
                        and call.text(1) == str(manager.ledger_path)):
                    try:
                        resource.setrlimit(resource.RLIMIT_NPROC, (0, original[1]))
                        observation['lowered'] = True
                    except OSError as error:
                        observation['callback_error'] = str(error)
                return None

            hold_system_calls(['openat'], decide)
            try:
                job = manager.submit(kind='analysis', owner='operator', request_id=None,
                    task=lambda context: {})
                terminal = manager.wait(job.job_id, 20.0)
            finally:
                resource.setrlimit(resource.RLIMIT_NPROC, original)
            snapshot = manager.status()
            retry = manager.submit(kind='analysis', owner='operator', request_id=None,
                task=lambda context: {'restored': True})
            recovered = manager.wait(retry.job_id, 20.0)
            finish({'observation': observation, 'status': terminal.status,
                'private': terminal.error, 'public': terminal.to_public_dict()['error'],
                'active': snapshot.active_count, 'unreaped': list(manager.unreaped_workers),
                'retry': recovered.status, 'result': recovered.result})
            """
        ),
        arguments=(str(tmp_path / "private-jobs"),),
    )
    assert result["observation"] == {"lowered": True, "callback_error": None}
    assert result["status"] == "failed"
    assert result["private"] == "Studio worker could not start: can't start new thread"
    assert result["public"] == "Studio worker could not start."
    assert result["active"] == 0 and result["unreaped"] == []
    assert result["retry"] == "completed" and result["result"] == {"restored": True}


def test_kernel_refused_group_signals_keep_an_honest_public_outcome(tmp_path: Path) -> None:
    """A real surviving descendant keeps its capacity and receives scoped cleanup."""
    result = run_child(
        dedent(
            """
            import errno, os, signal, sys, time
            from pathlib import Path
            from sc_neurocore.studio.platform.jobs import StudioJobManager
            from sc_neurocore.studio.platform.jobs_admission import StudioJobQueueFull
            from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
            from sc_neurocore.studio.platform.jobs_process_state import group_survivors
            from tests.studio_seccomp_support import Refusal, install_refusals
            from tests.studio_syscall_support import finish

            root = Path(sys.argv[1])
            manager = StudioJobManager(root=root, allowed_kinds=frozenset({'analysis'}),
                default_timeout_seconds=15.0, max_concurrent_jobs=1, max_queued_jobs=0)
            install_refusals([Refusal('kill', errno.EPERM)])
            submitted = manager.submit_process_task(kind='analysis', owner='operator',
                request_id=None, payload={}, task_path=
                'tests.test_studio_jobs_capacity_custody:task_leaving_descendant')
            terminal = manager.wait(submitted.job_id, 30.0)
            ledger = StudioJobLedger(root=root)
            row = ledger.connection().execute(
                'SELECT group_id FROM job_workers WHERE job_id=?',
                (submitted.job_id,)).fetchone()
            group = int(row['group_id'])
            assert group > 1 and group != os.getpgrp()
            try:
                assert int((root / submitted.job_id / 'group.pid').read_text()) == group
                snapshot = manager.status()
                observer = StudioJobManager(root=root,
                    allowed_kinds=frozenset({'analysis'}), default_timeout_seconds=15.0,
                    max_concurrent_jobs=1, max_queued_jobs=0)
                refused = False
                try:
                    observer.submit(kind='analysis', owner='observer', request_id=None,
                        task=lambda context: {})
                except StudioJobQueueFull:
                    refused = True
                outcome = {'status': terminal.status, 'private': terminal.error,
                    'public': terminal.to_public_dict()['error'],
                    'alive': bool(group_survivors(group)),
                    'unreaped': snapshot.unreaped_workers == (submitted.job_id,),
                    'running': snapshot.admission['running'],
                    'observer_running': observer.status().admission['running'],
                    'refused': refused}
            finally:
                for pid in group_survivors(group):
                    descriptor = os.pidfd_open(pid)
                    try:
                        signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                    finally:
                        os.close(descriptor)
                deadline = time.monotonic() + 10.0
                while group_survivors(group) and time.monotonic() < deadline:
                    time.sleep(0.01)
                ledger.close()
            outcome['cleaned'] = not group_survivors(group)
            finish(outcome)
            """
        ),
        arguments=(str(tmp_path / "private-jobs"),),
    )
    assert result["status"] == "failed"
    assert isinstance(result["public"], str) and "not reaped" in result["public"]
    assert result["private"] == result["public"] and str(tmp_path) not in result["public"]
    assert result["alive"] is True and result["unreaped"] is True
    assert result["running"] == 1 and result["observer_running"] == 1
    assert result["refused"] is True and result["cleaned"] is True
