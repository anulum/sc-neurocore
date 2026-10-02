# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real generation registration refusal exchanges

"""Preserve registration custody across real authority refusal and reply loss."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import secrets
import subprocess
import sys

import pytest

from sc_neurocore.studio.platform.jobs_failures import public_job_error
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.storage_finish_protocol import (
    FINISH_SCHEMA_VERSION,
    StorageFinishRequest,
)
from sc_neurocore.studio.platform.storage_generation_exchanges import (
    GenerationExchanges,
    StartRefused,
)
from tests.studio_storage_generation_runs import (
    api_runtime,
    authority as authority,
    base as base,
    ledger as ledger,
    reservations,
    workers,
)
from tests.studio_storage_generation_support import Authority
from tests.studio_storage_supervision_support import JOB, admit


@contextmanager
def live_worker() -> Iterator[subprocess.Popen[bytes]]:
    """Own a real process group throughout registration and reap it on exit."""
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)


def test_registration_refuses_the_real_foreign_lease_without_changing_custody(
    base: Path, ledger: StudioJobLedger, authority: Authority
) -> None:
    """A verified API peer cannot register a worker for another live supervisor."""
    with live_worker() as worker:
        admit(ledger, supervisor=supervisor_identity(worker.pid))
        before = ledger.record(JOB)
        exchange = GenerationExchanges(
            api_runtime(base, authority), job_id=JOB, generation=secrets.token_hex(16)
        )
        with pytest.raises(StartRefused) as caught:
            exchange.register(supervisor_identity(worker.pid))
        assert caught.value.outcome == "failed"
        assert caught.value.error == "Studio worker could not start: not_owner."
        assert public_job_error(caught.value.error) == caught.value.error
        assert ledger.record(JOB) == before
        assert workers(ledger) == 0
        assert reservations(ledger) == ["running"]
        assert authority.seen == ["start"]


def test_unanswered_registration_retains_capacity_until_actual_worker_reaping(
    base: Path, ledger: StudioJobLedger, authority: Authority
) -> None:
    """Three real lost replies preserve one registered worker and its reservation."""
    admit(ledger, supervisor=supervisor_identity())
    authority.lose["start"] = 3
    exchange = GenerationExchanges(
        api_runtime(base, authority), job_id=JOB, generation=secrets.token_hex(16)
    )
    with live_worker() as worker:
        with pytest.raises(StartRefused) as caught:
            exchange.register(supervisor_identity(worker.pid))
        error = caught.value.error
        assert caught.value.outcome == "failed"
        assert error == "Studio worker could not start: registration unanswered."
        assert public_job_error(error) == error
        assert ledger.record(JOB).status == "running"
        assert workers(ledger) == 1
        assert reservations(ledger) == ["running"]
        assert worker.poll() is None
        assert authority.seen == ["start", "start", "start"]
        worker.terminate()
        assert worker.wait(timeout=5.0) < 0
    reply = exchange.finish(
        StorageFinishRequest(
            schema_version=FINISH_SCHEMA_VERSION,
            operation="finish",
            request_id=secrets.token_hex(16),
            workspace="default",
            job_id=JOB,
            outcome="failed",
            result=None,
            error=error,
            public_error=public_job_error(error),
            artifacts=(),
            worker_reaped=True,
        ),
        (),
    )
    assert reply.reply == "sealed"
    record = ledger.record(JOB)
    assert (record.status, record.error, record.public_error) == ("failed", error, error)
    assert reservations(ledger) == []
    assert authority.seen == ["start", "start", "start", "finish"]
