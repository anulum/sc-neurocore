# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — laboratory cancellation custody under real reply loss

"""Exercise lost authority replies with real sockets, ledger and fitting workers."""

from __future__ import annotations

import time

import pytest

from sc_neurocore.studio.api.fits import FitRequest, fit_problem
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.policy_models import Principal
from sc_neurocore.studio.platform.storage_isolated_jobs import IsolatedJobManager
from sc_neurocore.studio.platform.storage_requester import delegated
from tests.studio_storage_generation_support import Authority
from tests.test_studio_fits_routes import _body

pytest_plugins = ("tests.studio_storage_generation_runs", "tests.test_studio_storage_isolated_jobs")


@pytest.mark.parametrize("route", ["/api/training/stop", "/api/fits/jobs/{job_id}/cancel"])
@pytest.mark.parametrize("roles", [frozenset(), frozenset({"studio.admin"})])
def test_denied_cancel_reply_loss_preserves_actor_job(
    manager: IsolatedJobManager,
    authority: Authority,
    route: str,
    roles: frozenset[str],
) -> None:
    """Losing a real denial cannot signal another actor's laboratory worker.

    Both legacy and laboratory cancellation routes retain admitted actor
    custody, including callers with the service administrator role. Cleanup
    uses the actual owner's acknowledged cancellation even if an assertion
    fails, so the regression leaves no fitting worker running.
    """
    alice = Principal("alice", frozenset())
    bob = Principal("bob", roles)
    fit = fit_problem(FitRequest.model_validate(_body(generations=2, population=4)))
    document = {"problem": fit.to_public_dict(), "generations": 500, "population": 100}
    with delegated(alice, method="POST", route="/api/fits/jobs", request_id="loss-submit"):
        job = manager.submit_process_task(
            kind="analysis",
            owner="alice",
            request_id=None,
            task_path="sc_neurocore.studio.api.fit_jobs:execute_laboratory_task",
            payload={"operation": "fit", "document": document},
            admission={"laboratory_task": "fit"},
        )
    try:
        with delegated(alice, method="GET", route="/api/fits/jobs/{job_id}", request_id="ready"):
            deadline = time.monotonic() + 30
            while manager.record(job.job_id).status != "running":
                assert time.monotonic() < deadline, "Fitting worker did not start."
                time.sleep(0.02)
        # Drop the actual reply after the production authority has denied Bob.
        authority.lose["cancel"] = 1
        with (
            delegated(bob, method="POST", route=route, request_id="loss-stop"),
            pytest.raises(EOFError),
        ):
            manager.cancel(job.job_id)
        with delegated(alice, method="GET", route="/api/fits/jobs/{job_id}", request_id="observe"):
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                record = manager.record(job.job_id)
                assert record.owner == "alice"
                assert record.status == "running", "Denied cancellation stopped the fitting worker."
                time.sleep(0.05)
    finally:
        with delegated(
            alice, method="POST", route="/api/fits/jobs/{job_id}/cancel", request_id="cleanup"
        ):
            manager.cancel(job.job_id)
            assert manager.wait(job.job_id, timeout_seconds=30).status == "cancelled"
    assert manager.generation_failures == {}


def test_service_cancel_reply_loss_retains_legacy_stop(
    manager: IsolatedJobManager,
    authority: Authority,
) -> None:
    """An admitted service task retains cancellation after an actual lost reply."""
    from tests.studio_storage_generation_runs import LONG

    operator = Principal("operator", frozenset({"studio.admin"}))
    with delegated(operator, method="POST", route="/api/analysis/jobs", request_id="service-start"):
        job = manager.submit_process_task(
            kind="analysis",
            owner="studio",
            request_id=None,
            task_path="sc_neurocore.studio.api.analysis_jobs:execute_analysis_process_task",
            payload=LONG,
        )
    try:
        with delegated(
            operator, method="GET", route="/api/studio/jobs/{job_id}", request_id="ready"
        ):
            deadline = time.monotonic() + 30
            while manager.record(job.job_id).status != "running":
                assert time.monotonic() < deadline, "Service worker did not start."
                time.sleep(0.02)
        authority.lose["cancel"] = 1
        with (
            delegated(operator, method="POST", route="/api/training/stop", request_id="lost-stop"),
            pytest.raises(EOFError),
        ):
            manager.cancel(job.job_id)
        with delegated(
            operator, method="GET", route="/api/studio/jobs/{job_id}", request_id="finish"
        ):
            assert manager.wait(job.job_id, timeout_seconds=30).status == "cancelled"
    finally:
        with delegated(operator, method="POST", route="/api/training/stop", request_id="cleanup"):
            manager.cancel(job.job_id)
            manager.wait(job.job_id, timeout_seconds=30)
    assert manager.generation_failures == {}


def test_acknowledged_cancel_of_recovered_job_has_no_local_worker(
    manager: IsolatedJobManager,
    ledger: StudioJobLedger,
) -> None:
    """A terminal job from another generation remains terminal after acknowledged stop."""
    from tests.studio_storage_finish_support import finish, request, started, stop
    from tests.studio_storage_supervision_support import JOB

    stop(started(ledger))
    assert finish(ledger, request({}), []).reply == "sealed"
    operator = Principal("operator", frozenset({"studio.admin"}))
    with delegated(
        operator, method="POST", route="/api/training/stop", request_id="recovered-stop"
    ):
        assert manager.cancel(JOB).status == "completed"
    assert manager.generation_failures == {}
