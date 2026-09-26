# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — isolated laboratory execution and actor custody

"""Run scientific tasks with the real authority, launcher and process worker.

Same-UID sockets establish functional isolation wiring, not OS privilege
separation. The authority itself must withhold other actors' records and
refuse their cancellations before mutation.
"""

from __future__ import annotations

import pytest

from sc_neurocore.studio.api.fits import FitRequest, fit_problem
from sc_neurocore.studio.platform.jobs_models import StudioJobRejected
from sc_neurocore.studio.platform.policy_models import Principal
from sc_neurocore.studio.platform.storage_isolated_jobs import IsolatedJobManager
from sc_neurocore.studio.platform.storage_requester import delegated
from tests.test_studio_fits_routes import _body
from tests.test_fitting_cohorts import cohort

pytest_plugins = ("tests.studio_storage_generation_runs", "tests.test_studio_storage_isolated_jobs")

TASK = "sc_neurocore.studio.api.fit_jobs:execute_laboratory_task"
ALICE = Principal("alice", frozenset())
BOB = Principal("bob", frozenset())


@pytest.mark.parametrize("operation", ["fit", "fit_replay", "cohort"])
def test_actor_owned_laboratory_execution_and_authority_custody(
    manager: IsolatedJobManager, operation: str
) -> None:
    """Each registered route launches science and hides its actor's job from peers."""
    fit = fit_problem(FitRequest.model_validate(_body(generations=2, population=4)))
    document = {"problem": fit.to_public_dict(), "generations": 2, "population": 4}
    route = "/api/fits/jobs"
    if operation == "cohort":
        document = cohort().to_public_dict()
        route = "/api/cohorts/jobs"
    elif operation == "fit_replay":
        from sc_neurocore.fitting.fit import fit_parameters

        document = fit_parameters(fit, generations=2, population=4)
        route = "/api/fits/replay/jobs"
    with delegated(ALICE, method="POST", route=route, request_id="laboratory-test"):
        with pytest.raises(StudioJobRejected, match="owner"):
            manager.submit_process_task(
                kind="analysis",
                owner="bob",
                request_id=None,
                task_path=TASK,
                payload={"operation": operation, "document": document},
            )
        submitted = manager.submit_process_task(
            kind="analysis",
            owner="alice",
            request_id=None,
            task_path=TASK,
            payload={"operation": operation, "document": document},
            admission={"laboratory_task": operation},
        )
        completed = manager.wait(submitted.job_id, timeout_seconds=60)
    assert completed.status == "completed", completed.error
    assert completed.owner == "alice"
    assert completed.result is not None
    assert {artifact.relative_path for artifact in completed.artifacts} == {
        "history.jsonl",
        "experiment.json",
        "result.json",
    }
    if operation == "fit_replay":
        assert completed.result["reproduced"] is True
    for actor in (BOB, ALICE):
        with delegated(actor, method="GET", route="/api/fits/jobs/{job_id}", request_id="read"):
            if actor == BOB:
                with pytest.raises(KeyError):
                    manager.record(submitted.job_id)
            else:
                assert manager.record(submitted.job_id) == completed
        with delegated(
            actor, method="POST", route="/api/fits/jobs/{job_id}/cancel", request_id="stop"
        ):
            if actor == BOB:
                with pytest.raises(KeyError):
                    manager.cancel(submitted.job_id)
            else:
                assert manager.cancel(submitted.job_id) == completed
    assert manager.generation_failures == {}


def test_denied_actor_cancellation_does_not_stop_a_live_worker(
    manager: IsolatedJobManager,
) -> None:
    """A refused authority mutation cannot signal the local generation's cancellation."""
    import time

    fit = fit_problem(FitRequest.model_validate(_body(generations=2, population=4)))
    document = {"problem": fit.to_public_dict(), "generations": 500, "population": 100}
    with delegated(ALICE, method="POST", route="/api/fits/jobs", request_id="large-fit"):
        job = manager.submit_process_task(
            kind="analysis",
            owner="alice",
            request_id=None,
            task_path=TASK,
            payload={"operation": "fit", "document": document},
            admission={"laboratory_task": "fit"},
        )
        deadline = time.monotonic() + 30
        while manager.record(job.job_id).status != "running":
            assert time.monotonic() < deadline
            time.sleep(0.02)
    with (
        delegated(
            BOB, method="POST", route="/api/fits/jobs/{job_id}/cancel", request_id="deny-stop"
        ),
        pytest.raises(KeyError),
    ):
        manager.cancel(job.job_id)
    with delegated(ALICE, method="GET", route="/api/fits/jobs/{job_id}", request_id="observe"):
        # Give the supervisor several polling intervals to react to any illegal signal.
        time.sleep(0.3)
        assert manager.record(job.job_id).status == "running"
    with delegated(
        ALICE, method="POST", route="/api/fits/jobs/{job_id}/cancel", request_id="owner-stop"
    ):
        assert manager.cancel(job.job_id).status in {"cancelling", "cancelled"}
        assert manager.wait(job.job_id, timeout_seconds=30).status == "cancelled"
    assert manager.generation_failures == {}


@pytest.mark.parametrize(
    "operation,admission",
    [
        ("cohort", {"laboratory_task": "cohort"}),
        ("fit", None),
        ("fit", {"laboratory_task": "cohort"}),
    ],
)
def test_inconsistent_laboratory_admission_refuses_before_allocation(
    manager: IsolatedJobManager,
    operation: str,
    admission: dict[str, object] | None,
) -> None:
    """Route, operation and custody marker cannot disagree at named admission."""
    with (
        delegated(ALICE, method="POST", route="/api/fits/jobs", request_id="bad-custody"),
        pytest.raises(ValueError, match="operation and admission"),
    ):
        manager.submit_process_task(
            kind="analysis",
            owner="alice",
            request_id=None,
            task_path=TASK,
            payload={"operation": operation, "document": {}},
            admission=admission,
        )
    operator = Principal("operator", frozenset({"studio.admin"}))
    with delegated(operator, method="GET", route="/api/studio/jobs", request_id="verify-empty"):
        assert manager.list_records() == ()
    assert manager.generation_failures == {}


def test_http_security_and_scientific_routes_use_real_isolated_manager(
    manager: IsolatedJobManager,
) -> None:
    """Public HTTP composition carries security delegation through isolated execution.

    This composes production routers and middleware with the real isolated
    collaborator. It does not qualify the distinct-UID startup preflight.
    """
    from fastapi import FastAPI
    from starlette.testclient import TestClient
    from sc_neurocore.studio.api.runtime import build_studio_api_context
    from sc_neurocore.studio.api.security import install_studio_security_middleware
    from sc_neurocore.studio.api.fits import build_fits_router
    from sc_neurocore.studio.platform.settings import StudioRuntimeSettings
    from tests.test_studio_fit_jobs import _finish

    application = FastAPI()
    context = build_studio_api_context(
        application,
        StudioRuntimeSettings(enforce_route_policies=True),
    )
    context.studio_job_manager = manager
    application.state.studio_job_manager = manager
    install_studio_security_middleware(application, context)
    application.include_router(build_fits_router(context))
    with TestClient(
        application, base_url="http://127.0.0.1", headers={"x-studio-principal": "alice"}
    ) as client:
        submitted = client.post("/api/fits/jobs", json=_body(generations=2, population=4))
        assert submitted.status_code == 202, submitted.text
        route = submitted.json()["status_route"]
        assert client.get(route, headers={"x-studio-principal": "bob"}).status_code == 404
        assert (
            client.post(route + "/cancel", headers={"x-studio-principal": "bob"}).status_code == 404
        )
        completed = _finish(client, route)
        assert completed["owner"] == "alice"
        assert completed["execution_model"] == "process"
        assert completed["result"]["result_sha256"]


@pytest.mark.parametrize(
    "operation,admission",
    [
        ("cohort", {"laboratory_task": "cohort"}),
        ("fit", None),
        ("fit", {"laboratory_task": "cohort"}),
    ],
)
def test_storage_authority_rejects_inconsistent_wire_admission(
    operation: str,
    admission: dict[str, object] | None,
) -> None:
    """The real peer-bound preparer refuses inconsistent wire bytes independently."""
    from sc_neurocore.studio.platform.policy_models import InMemoryAuditSink
    from tests.test_studio_storage_named_admission import _exchange, _request

    request = _request()
    request.update(
        task_name="laboratory.run",
        authorized_route="/api/fits/jobs",
        requester={"principal_id": "alice", "roles": []},
        payload={"operation": operation, "document": {}},
        admission=admission,
        seed_manifest={},
    )
    with pytest.raises(ValueError, match="operation and admission"):
        _exchange(request, {}, sink=InMemoryAuditSink())


def test_requester_owned_task_cannot_run_for_anonymous_delegation(
    manager: IsolatedJobManager,
) -> None:
    """A delegated request without identity cannot acquire laboratory actor custody."""
    with (
        delegated(None, method="POST", route="/api/fits/jobs", request_id="anonymous"),
        pytest.raises(ValueError, match="requires authenticated identity"),
    ):
        manager.submit_process_task(
            kind="analysis",
            owner="local",
            request_id=None,
            task_path=TASK,
            payload={"operation": "fit", "document": {}},
            admission={"laboratory_task": "fit"},
        )
    operator = Principal("operator", frozenset({"studio.admin"}))
    with delegated(operator, method="GET", route="/api/studio/jobs", request_id="no-job"):
        assert manager.list_records() == ()
