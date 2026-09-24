# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Managed worker startup custody

"""Real subprocess registration must precede even task module import."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_process_protocol import _process_worker_environment
from sc_neurocore.studio.platform.jobs_shared_admission import SharedJobAdmission
from sc_neurocore.studio.platform.jobs_reaper import reap_process_group
from sc_neurocore.studio.platform.jobs_worker_registration import (
    await_worker_registration,
    start_worker_registration,
)


@pytest.mark.parametrize(
    "case",
    [
        "valid",
        "no-worker-db",
        "grant-eof",
        "grant-malformed",
        "grant-partial",
        "grant-trailing",
        "grant-silent",
        "missing",
        "wrong-supervisor",
        "cancelling",
        "duplicate",
        "no-session",
    ],
)
def test_managed_worker_registers_before_task_import(tmp_path: Path, case: str) -> None:
    """Invalid custody produces failure evidence without importing or running the task."""
    root = tmp_path / "jobs"
    ledger = StudioJobLedger(root=root)
    controller = SharedJobAdmission(ledger, max_concurrent=1, max_queued=0)
    job_id = "sj_0000000000000001"
    controller.admit(
        job_id=job_id,
        kind="analysis",
        actor="owner",
        workspace="default",
        request_id=None,
        idempotency_key=None,
        experiment_sha256=None,
        admission=None,
        execution_model="process",
    )
    ledger.transition(job_id, "running")
    supervisor = ledger.supervisor
    if case == "cancelling":
        ledger.transition(job_id, "cancelling")
    elif case == "wrong-supervisor":
        supervisor = "unknown-host:1:unknown"
    with ledger.transaction() as connection:
        if case == "missing":
            connection.execute("DELETE FROM admission_reservations WHERE job_id=?", (job_id,))
        if case == "duplicate":
            connection.execute(
                "INSERT INTO job_workers VALUES(?,?,?,?,?)",
                (job_id, ledger.supervisor, "preserved-identity", "preserved-boot", 1),
            )
    before = ledger.record(job_id)
    work_dir = root / job_id
    work_dir.mkdir()
    payload, result = work_dir / "payload.json", work_dir / "result.json"
    payload.write_text("{}")
    module = tmp_path / "registration_probe.py"
    module.write_text(
        "from pathlib import Path\n"
        "Path(__file__).with_suffix('.imported').write_text('imported')\n"
        "def run(context,payload):\n"
        "    Path(__file__).with_suffix('.executed').write_text('executed')\n"
        "    return {'actual_execution': True}\n"
    )
    environment = _process_worker_environment()
    environment["PYTHONPATH"] = str(tmp_path) + os.pathsep + environment["PYTHONPATH"]
    entrypoint = ["-m", "sc_neurocore.studio.platform.process_worker"]
    if case == "no-worker-db":
        # The real entry point runs unchanged; an audit hook only records any
        # database connection the worker opens and reports it on stderr.
        entrypoint = [
            "-c",
            "import sys\n"
            "opened = []\n"
            "sys.addaudithook(lambda event, args: opened.append(event)"
            " if event == 'sqlite3.connect' else None)\n"
            "from sc_neurocore.studio.platform.process_worker import main\n"
            "code = main()\n"
            "print(f'sqlite3.connect events: {len(opened)}', file=sys.stderr)\n"
            "raise SystemExit(code)\n",
        ]
    child: subprocess.Popen[bytes] | None = None
    try:
        child = subprocess.Popen(
            [
                sys.executable,
                *entrypoint,
                "--task",
                "registration_probe:run",
                "--payload",
                str(payload),
                "--result",
                str(result),
                "--work-dir",
                str(work_dir),
                "--max-artifact-bytes",
                "1024",
                "--supervisor",
                supervisor,
            ],
            env=environment,
            start_new_session=case != "no-session",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
        )
        if case.startswith("grant-"):
            assert child.stdin is not None
            if case == "grant-silent":
                child.wait(timeout=5.0)
            elif case != "grant-eof":
                token = {"grant-partial": b"rea", "grant-trailing": b"ready\nx"}.get(case, b"bad")
                child.stdin.write(token)
                child.stdin.flush()
            child.stdin.close()
        else:
            registration = start_worker_registration(ledger, job_id, child, supervisor)
            registration.join(timeout=5.0)
            assert not registration.is_alive()
        # The registration sender owns stdin and has already closed it.
        child.stdin = None
        _, stderr = child.communicate(timeout=5.0)
        valid = case in {"valid", "no-worker-db"}
        assert child.returncode == (0 if valid else 1), stderr
        if case == "no-worker-db":
            assert b"sqlite3.connect events: 0" in stderr
        evidence = json.loads(result.read_text())
        assert evidence["status"] == ("completed" if valid else "failed")
        assert module.with_suffix(".imported").exists() is valid
        assert module.with_suffix(".executed").exists() is valid
        rows = ledger.connection().execute("SELECT * FROM job_workers").fetchall()
        if valid:
            assert len(rows) == 1 and rows[0]["supervisor"] == ledger.supervisor
            assert evidence["result"] == {"actual_execution": True}
        elif case == "duplicate":
            assert len(rows) == 1 and rows[0]["worker_identity"] == "preserved-identity"
        else:
            assert rows == []
        assert ledger.record(job_id) == before
    finally:
        try:
            if child is not None:
                if case != "no-session":
                    assert reap_process_group(child, owned_group_id=child.pid).reaped
                else:
                    if child.poll() is None:
                        child.kill()
                    child.wait(timeout=3.0)
                if child.stdout is not None:
                    child.stdout.close()
                if child.stderr is not None:
                    child.stderr.close()
        finally:
            ledger.close()


@pytest.mark.parametrize(
    "fragments,accepted",
    [((b"rea", b"dy\n"), True), ((b"rea",), False), ((b"ready\nx",), False), ((), False)],
)
def test_registration_pipe_requires_complete_grant_and_eof(
    fragments: tuple[bytes, ...], accepted: bool
) -> None:
    """Accept a fragmented grant only after EOF; refuse truncation and trailing bytes."""
    reader, writer = os.pipe()
    finished = threading.Event()
    failures: list[BaseException] = []

    def receive() -> None:
        try:
            await_worker_registration(reader)
        except BaseException as exc:
            failures.append(exc)
        finally:
            os.close(reader)
            finished.set()

    receiver = threading.Thread(target=receive)
    receiver.start()
    writer_open = True
    try:
        for fragment in fragments:
            os.write(writer, fragment)
            if accepted:
                assert not finished.wait(0.05), "A grant without EOF authorised execution"
        os.close(writer)
        writer_open = False
        assert finished.wait(4.0)
        assert (not failures) is accepted
        if failures:
            assert isinstance(failures[0], RuntimeError)
            assert str(failures[0]) == "Worker registration was not confirmed."
    finally:
        if writer_open:
            os.close(writer)
        receiver.join(timeout=4.0)
        assert not receiver.is_alive()


def test_registration_without_private_pipe_preserves_ledger(tmp_path: Path) -> None:
    """Refuse an actual process lacking the private grant channel before database writes."""
    ledger = StudioJobLedger(root=tmp_path / "jobs")
    child = subprocess.Popen([sys.executable, "-c", "pass"], stdin=subprocess.DEVNULL)
    try:
        before = ledger.connection().total_changes
        with pytest.raises(ValueError, match="private registration pipe"):
            start_worker_registration(ledger, "sj_0000000000000001", child, ledger.supervisor)
        assert ledger.connection().total_changes == before
        assert ledger.connection().execute("SELECT COUNT(*) FROM job_workers").fetchone()[0] == 0
        assert child.wait(timeout=3.0) == 0
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=3.0)
        ledger.close()


def test_registration_silent_pipe_times_out_without_closing_callers_descriptor() -> None:
    """A connected but silent sender cannot hold registration indefinitely."""
    reader, writer = os.pipe()
    try:
        with pytest.raises(RuntimeError, match="Worker registration timed out"):
            await_worker_registration(reader)
        assert os.fstat(reader).st_ino == os.fstat(writer).st_ino
    finally:
        os.close(reader)
        os.close(writer)
