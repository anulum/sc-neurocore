# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — delegated job lease ownership

"""A storage service may renew a lease only for the delegated owner it verified."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_writes import heartbeat_job, transition_job
from sc_neurocore.studio.platform.jobs_models import StudioJobRejected

_JOB = "sj_" + "9" * 16
_API = "api-host:4242:123456"


class Clock:
    """Controllable UTC clock for second-resolution lease timestamps."""

    def __init__(self) -> None:
        self.now = datetime(2026, 9, 24, 0, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now


def _ledger(tmp_path: Path, clock: Clock) -> StudioJobLedger:
    ledger = StudioJobLedger(root=tmp_path, supervisor="storage:1:1", clock=clock)
    ledger.create(
        job_id=_JOB,
        kind="analysis",
        actor="operator",
        workspace="default",
        request_id=None,
        idempotency_key=None,
        experiment_sha256=None,
        admission=None,
        execution_model="process",
    )
    with ledger.transaction() as connection:
        connection.execute("UPDATE jobs SET lease_owner = ? WHERE job_id = ?", (_API, _JOB))
    return ledger


def _lease(ledger: StudioJobLedger) -> tuple[str, str, str]:
    row = (
        ledger.connection()
        .execute(
            "SELECT lease_owner, lease_expires_at_utc, heartbeat_at_utc FROM jobs WHERE job_id=?",
            (_JOB,),
        )
        .fetchone()
    )
    return str(row[0]), str(row[1]), str(row[2])


def test_delegated_transition_renews_the_delegated_lease(tmp_path: Path) -> None:
    """A transition acting for the recorded owner renews that owner's lease."""
    clock = Clock()
    ledger = _ledger(tmp_path, clock)
    try:
        before = _lease(ledger)
        clock.now += timedelta(seconds=7)
        transition_job(ledger, _JOB, "running", supervisor=_API)
        after = _lease(ledger)
        assert after[0] == _API
        assert after[1] > before[1] and after[2] > before[2]
    finally:
        ledger.close()


@pytest.mark.parametrize("acting", [None, "api-host:9:9"])
def test_transition_by_another_supervisor_keeps_the_lease(
    tmp_path: Path, acting: str | None
) -> None:
    """The service's own or a foreign supervisor never renews a delegated lease."""
    clock = Clock()
    ledger = _ledger(tmp_path, clock)
    try:
        before = _lease(ledger)
        clock.now += timedelta(seconds=7)
        record = transition_job(ledger, _JOB, "running", supervisor=acting)
        assert record.status == "running"
        assert _lease(ledger) == before
    finally:
        ledger.close()


def test_delegated_heartbeat_renews_only_the_owner(tmp_path: Path) -> None:
    """The delegated owner's heartbeat renews; any other supervisor is refused unchanged."""
    clock = Clock()
    ledger = _ledger(tmp_path, clock)
    try:
        before = _lease(ledger)
        clock.now += timedelta(seconds=7)
        for acting in (None, "api-host:9:9"):
            with pytest.raises(StudioJobRejected, match="another supervisor"):
                heartbeat_job(ledger, _JOB, supervisor=acting)
            assert _lease(ledger) == before
        assert heartbeat_job(ledger, _JOB, supervisor=_API) is True
        after = _lease(ledger)
        assert after[0] == _API and after[1] > before[1] and after[2] > before[2]
    finally:
        ledger.close()


def test_terminal_job_heartbeat_is_a_no_op_for_the_owner(tmp_path: Path) -> None:
    """A finished job acquires no new lease, even from its delegated owner."""
    clock = Clock()
    ledger = _ledger(tmp_path, clock)
    try:
        transition_job(ledger, _JOB, "failed", supervisor=_API, error="stopped")
        before = _lease(ledger)
        clock.now += timedelta(seconds=7)
        assert heartbeat_job(ledger, _JOB, supervisor=_API) is False
        assert _lease(ledger) == before
    finally:
        ledger.close()
