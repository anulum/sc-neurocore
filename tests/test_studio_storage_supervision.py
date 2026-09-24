# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — delegated job supervision authority

"""Start and heartbeat act only for the verified API generation that owns a job.

The storage handler and API client exchange real frames over a connected Unix
socket pair; the ledger is a real SQLite authority whose own supervisor
differs from the delegated API generation, as in the isolated profile.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from datetime import timedelta
from typing import cast

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_models import StudioJobStatus
from sc_neurocore.studio.platform.storage_peer import read_verified_frame, write_verified_frame
from sc_neurocore.studio.platform.storage_supervision import serve_supervision
from sc_neurocore.studio.platform.storage_supervision_protocol import (
    decode_supervision_response,
    encode_supervision_message,
)
from tests.studio_storage_supervision_support import *


def test_start_marks_running_and_registers_the_worker(ledger: StudioJobLedger, worker: str) -> None:
    """The owning API generation starts its job and binds the verified worker."""
    api = supervisor_identity()
    admit(ledger, supervisor=api)
    response = exchange(ledger, start(worker))
    assert (response.outcome, response.reason) == ("started", None)
    record = ledger.record(JOB)
    assert record.status == "running" and record.started_at_utc is not None
    assert worker_rows(ledger) == [(api, worker)]
    retried = exchange(ledger, start(worker))
    assert retried.outcome == "started"
    assert worker_rows(ledger) == [(api, worker)]


def test_second_worker_for_a_started_job_is_a_conflict(
    ledger: StudioJobLedger, worker: str
) -> None:
    """A different worker identity can never replace the registered one."""
    admit(ledger, supervisor=supervisor_identity())
    assert exchange(ledger, start(worker)).outcome == "started"
    host, _, token = worker.split(":")
    other = f"{host}:{os.getpid()}:{token}"
    response = exchange(ledger, start(other))
    assert (response.outcome, response.reason) == ("refused", "worker_conflict")
    assert [row[1] for row in worker_rows(ledger)] == [worker]


def test_cancelled_before_start_keeps_cancellation_and_withholds_worker(
    ledger: StudioJobLedger, worker: str
) -> None:
    """A job cancelled before its worker starts is never granted a worker."""
    admit(ledger, supervisor=supervisor_identity())
    ledger.transition(JOB, "cancelling")
    response = exchange(ledger, start(worker))
    assert (response.outcome, response.reason) == ("cancelling", None)
    assert ledger.record(JOB).status == "cancelling"
    assert worker_rows(ledger) == []


@pytest.mark.parametrize("case", ["other-owner", "unknown-job", "other-workspace"])
def test_foreign_or_missing_jobs_are_refused_without_change(
    ledger: StudioJobLedger, worker: str, case: str
) -> None:
    """Only the delegated owner in the configured workspace can start a job.

    For ``other-owner`` the job is delegated to another live process generation.
    """
    if case == "other-owner":
        admit(ledger, supervisor=worker)
    elif case == "other-workspace":
        admit(ledger, supervisor=supervisor_identity(), workspace="elsewhere")
    response = exchange(ledger, start(worker))
    expected = "not_owner" if case == "other-owner" else "not_found"
    assert (response.outcome, response.reason) == ("refused", expected)
    if case != "unknown-job":
        assert ledger.record(JOB).status == "pending"
    assert worker_rows(ledger) == []


def test_dead_worker_is_refused_before_the_job_starts(ledger: StudioJobLedger) -> None:
    """An exited worker generation is refused and the job stays pending."""
    admit(ledger, supervisor=supervisor_identity())
    child = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    identity = supervisor_identity(child.pid)
    child.wait(timeout=10.0)
    response = exchange(ledger, start(identity))
    assert (response.outcome, response.reason) == ("refused", "worker_unverified")
    assert ledger.record(JOB).status == "pending"
    assert worker_rows(ledger) == []


@pytest.mark.parametrize("status", ["failed", "unknown"])
def test_ended_or_unknown_jobs_are_not_live(
    ledger: StudioJobLedger, worker: str, status: str
) -> None:
    """Terminal and unresolved jobs accept neither start nor heartbeat."""
    admit(ledger, supervisor=supervisor_identity())
    ledger.transition(JOB, cast(StudioJobStatus, status))
    for request in (start(worker), heartbeat()):
        response = exchange(ledger, request)
        assert (response.outcome, response.reason) == ("refused", "not_live")
    assert worker_rows(ledger) == []


def test_heartbeat_renews_the_delegated_owner_lease(ledger: StudioJobLedger, clock: Clock) -> None:
    """The owner's heartbeat extends the lease and keeps the delegated owner."""
    admit(ledger, supervisor=supervisor_identity())
    before = (
        ledger.connection()
        .execute("SELECT lease_expires_at_utc, lease_owner FROM jobs WHERE job_id=?", (JOB,))
        .fetchone()
    )
    clock.now += timedelta(seconds=5)
    response = exchange(ledger, heartbeat())
    assert (response.outcome, response.reason) == ("renewed", None)
    after = (
        ledger.connection()
        .execute("SELECT lease_expires_at_utc, lease_owner FROM jobs WHERE job_id=?", (JOB,))
        .fetchone()
    )
    assert after["lease_owner"] == before["lease_owner"] == supervisor_identity()
    assert after["lease_expires_at_utc"] > before["lease_expires_at_utc"]


def test_heartbeat_reports_a_recorded_cancellation(ledger: StudioJobLedger, clock: Clock) -> None:
    """The owner learns of a cancellation from its heartbeat; the lease still renews."""
    admit(ledger, supervisor=supervisor_identity())
    ledger.transition(JOB, "cancelling")
    before = ledger.record(JOB).lease_expires_at_utc
    clock.now += timedelta(seconds=5)
    response = exchange(ledger, heartbeat())
    assert (response.outcome, response.reason) == ("cancelling", None)
    record = ledger.record(JOB)
    assert record.status == "cancelling"
    assert before is not None and record.lease_expires_at_utc is not None
    assert record.lease_expires_at_utc > before


def test_listener_pre_read_frame_is_served_identically(
    ledger: StudioJobLedger, worker: str
) -> None:
    """A frame the listener already read for dispatch yields the same outcome."""
    admit(ledger, supervisor=supervisor_identity())
    service, client = socket.socketpair()
    request = start(worker)
    with service, client:
        deadline = time.monotonic() + 10.0
        write_verified_frame(
            client,
            encode_supervision_message(request),
            expected_uid=os.getuid(),
            max_bytes=4096,
            deadline=deadline,
        )
        first = read_verified_frame(
            service, expected_uid=os.getuid(), max_bytes=4096, deadline=deadline
        )
        serve_supervision(
            service,
            ledger=ledger,
            workspace="default",
            expected_api_uid=os.getuid(),
            max_bytes=4096,
            deadline=deadline,
            initial_frame=first,
        )
        reply = read_verified_frame(
            client, expected_uid=os.getuid(), max_bytes=4096, deadline=deadline
        )
    response = decode_supervision_response(reply, request=request, max_bytes=4096)
    assert (response.outcome, response.reason) == ("started", None)
