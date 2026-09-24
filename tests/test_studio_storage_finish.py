# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — delegated job finish authority

"""Finish seals exact bytes once, for the delegated owner of a stopped worker.

The authority and client exchange real frames over a socket pair; the ledger
is a real SQLite authority with delegated admission and a started job whose
worker is a real process. Races run a real competing writer on its own
connection between the ownership check and the terminal commit.
"""

from __future__ import annotations

import os
import socket
import stat
import time

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.storage_finish import serve_finish
from sc_neurocore.studio.platform.storage_finish_protocol import (
    encode_finish_message,
)
from sc_neurocore.studio.platform.storage_peer import write_verified_frame
from tests.studio_storage_finish_support import FILES, finish, request, reserved, started, stop
from tests.studio_storage_supervision_support import *


def test_stopped_worker_output_is_sealed_once(ledger: StudioJobLedger) -> None:
    """Exact bytes are sealed read-only, the record is terminal and capacity freed."""
    stop(started(ledger))
    sent = request(FILES)
    response = finish(ledger, sent, list(FILES.values()))
    assert (response.reply, response.reason) == ("sealed", None)
    record = ledger.record(JOB)
    assert record.status == "completed" and record.result == {"answer": 42}
    assert [artifact.relative_path for artifact in record.artifacts] == list(FILES)
    sealed = ledger.path.parent / JOB
    for name, payload in FILES.items():
        assert (sealed / name).read_bytes() == payload
        assert stat.S_IMODE((sealed / name).stat().st_mode) == 0o400
    assert reserved(ledger) == 0
    retry = finish(ledger, request(FILES, request_id="e" * 32), list(FILES.values()))
    assert (retry.reply, retry.reason) == ("already_sealed", None)
    changed = {**FILES, "weights.bin": b"\x09"}
    conflict = finish(ledger, request(changed), list(changed.values()))
    assert (conflict.reply, conflict.reason) == ("refused", "conflict")
    assert (sealed / "weights.bin").read_bytes() == b"\x00\x01"


def test_failed_outcome_is_recorded_with_its_error(ledger: StudioJobLedger) -> None:
    """An unsuccessful outcome seals what exists and keeps the error."""
    stop(started(ledger))
    response = finish(ledger, request({}, outcome="failed"), [])
    assert response.reply == "sealed"
    record = ledger.record(JOB)
    assert (record.status, record.error, record.artifacts) == ("failed", "worker failed", ())


def test_a_job_that_never_started_ends_unsuccessfully_only(ledger: StudioJobLedger) -> None:
    """A launch that failed before any worker is recorded; completion is refused."""
    ledger.path.parent.chmod(0o700)
    admit(ledger, supervisor=supervisor_identity())
    completed = finish(ledger, request(FILES), list(FILES.values()))
    assert (completed.reply, completed.reason) == ("refused", "not_live")
    response = finish(ledger, request({}, outcome="failed"), [])
    assert (response.reply, response.reason) == ("sealed", None)
    record = ledger.record(JOB)
    assert (record.status, record.error, record.started_at_utc) == ("failed", "worker failed", None)
    assert reserved(ledger) == 0


def test_an_unreaped_worker_keeps_its_capacity(ledger: StudioJobLedger) -> None:
    """Survivors the launcher could not stop keep the reservation, marked unreaped."""
    stop(started(ledger))
    response = finish(ledger, request({}, outcome="cancelled", worker_reaped=False), [])
    assert (response.reply, response.reason) == ("sealed", None)
    assert ledger.record(JOB).status == "cancelled"
    state = ledger.connection().execute("SELECT state FROM admission_reservations").fetchall()
    assert [row["state"] for row in state] == ["unreaped"]


def test_live_worker_is_never_finished(ledger: StudioJobLedger) -> None:
    """A registered worker that still runs refuses the finish before any byte moves."""
    worker = started(ledger)
    try:
        response = finish(ledger, request(FILES), list(FILES.values()))
        assert (response.reply, response.reason) == ("refused", "worker_live")
        assert ledger.record(JOB).status == "running"
        assert not (ledger.path.parent / JOB).exists()
        assert reserved(ledger) == 1
    finally:
        stop(worker)


@pytest.mark.parametrize("case", ["unknown", "other-owner", "not-started"])
def test_foreign_missing_or_unstarted_jobs_are_refused(
    ledger: StudioJobLedger, worker: str, case: str
) -> None:
    """Only the delegated owner of a started job may finish it."""
    ledger.path.parent.chmod(0o700)
    if case == "other-owner":
        admit(ledger, supervisor=worker)
        ledger.transition(JOB, "running")
    elif case == "not-started":
        admit(ledger, supervisor=supervisor_identity())
    response = finish(ledger, request(FILES), list(FILES.values()))
    expected = {"unknown": "not_found", "other-owner": "not_owner", "not-started": "not_live"}
    assert (response.reply, response.reason) == ("refused", expected[case])
    assert not (ledger.path.parent / JOB).exists()


def test_bytes_that_differ_from_the_manifest_are_not_sealed(ledger: StudioJobLedger) -> None:
    """A frame whose digest differs from its declaration seals nothing."""
    stop(started(ledger))
    declared = request(FILES)
    wrong = [b'{"ok": false}'[: len(FILES["reports/summary.json"])], *list(FILES.values())[1:]]
    response = finish(ledger, declared, wrong)
    assert (response.reply, response.reason) == ("refused", "bytes")
    assert ledger.record(JOB).status == "running"
    assert not (ledger.path.parent / JOB).exists()


def test_an_occupied_artefact_path_is_a_conflict(ledger: StudioJobLedger) -> None:
    """Bytes already sealed differently at a path are never replaced."""
    stop(started(ledger))
    job = ledger.path.parent / JOB
    job.mkdir(mode=0o700)
    (job / "weights.bin").write_bytes(b"other")
    response = finish(ledger, request(FILES), list(FILES.values()))
    assert (response.reply, response.reason) == ("refused", "conflict")
    assert (job / "weights.bin").read_bytes() == b"other"
    assert ledger.record(JOB).status == "running"


def test_job_ending_before_the_commit_is_answered_from_its_record(
    ledger: StudioJobLedger,
) -> None:
    """A job that became terminal after the check is never overwritten."""
    stop(started(ledger))

    def end_elsewhere() -> None:
        ledger.transition(JOB, "failed", error="reconciled")

    response = finish(ledger, request(FILES), list(FILES.values()), competing=end_elsewhere)
    assert (response.reply, response.reason) == ("refused", "conflict")
    assert ledger.record(JOB).status == "failed"


def _refused_before_reply(ledger: StudioJobLedger, payload: bytes, workspace: str) -> BaseException:
    service, client = socket.socketpair()
    with client:
        write_verified_frame(
            client, payload, expected_uid=os.getuid(), max_bytes=4096, deadline=time.monotonic() + 5
        )
        with pytest.raises(ValueError) as refused:
            serve_finish(
                service,
                ledger=ledger,
                workspace=workspace,
                expected_api_uid=os.getuid(),
                frame_max_bytes=4096,
                max_artifact_bytes=4,
                max_artifact_entries=16,
                deadline=time.monotonic() + 5,
            )
        client.settimeout(5.0)
        assert client.recv(16) == b""
    return refused.value


def test_invalid_requests_close_without_an_answer(ledger: StudioJobLedger) -> None:
    """Another workspace, an over-budget manifest or no workspace are refused silently."""
    foreign = encode_finish_message(request({}, workspace="elsewhere"))
    assert "configured workspace" in str(_refused_before_reply(ledger, foreign, "default"))
    budget = encode_finish_message(request(FILES))
    assert "aggregate limit" in str(_refused_before_reply(ledger, budget, "default"))
    service, client = socket.socketpair()
    with client, pytest.raises(ValueError, match="nonempty"):
        serve_finish(
            service,
            ledger=ledger,
            workspace="",
            expected_api_uid=os.getuid(),
            frame_max_bytes=4096,
            max_artifact_bytes=4,
            max_artifact_entries=16,
            deadline=time.monotonic() + 5,
        )
