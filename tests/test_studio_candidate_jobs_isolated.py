# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Candidate HTTP through the real isolated worker launcher

"""Qualify functional authority/launcher wiring without claiming UID separation."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from sc_neurocore.studio.api.candidate_jobs import build_candidate_jobs_router
from sc_neurocore.studio.api.candidates import build_candidates_router
from sc_neurocore.studio.api.runtime import build_studio_api_context
from sc_neurocore.studio.api.security import install_studio_security_middleware
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings
from sc_neurocore.studio.platform.policy_models import Principal
from sc_neurocore.studio.platform.storage_requester import delegated
from sc_neurocore.studio.platform.storage_isolated_jobs import IsolatedJobManager
from tests.studio_candidate_jobs_support import finish_candidate_job
from tests.studio_candidate_support import adex_candidate

pytest_plugins = ("tests.studio_storage_generation_runs", "tests.test_studio_storage_isolated_jobs")


@pytest.mark.parametrize("route", ["simulate", "review-packet"])
def test_candidate_http_runs_through_actual_authority_and_launcher(
    manager: IsolatedJobManager, route: str
) -> None:
    """Production routers carry requester identity into actual named process execution."""
    application = FastAPI()
    context = build_studio_api_context(
        application, StudioRuntimeSettings(enforce_route_policies=True)
    )
    context.studio_job_manager = manager
    application.state.studio_job_manager = manager
    install_studio_security_middleware(application, context)
    application.include_router(build_candidates_router(context))
    application.include_router(build_candidate_jobs_router(context))
    candidate = adex_candidate()
    candidate["reference_tests"][0]["steps"] = 2
    candidate["reference_tests"] = candidate["reference_tests"][:1]
    body = {"candidate": candidate, **({"steps": 2} if route == "simulate" else {})}
    with TestClient(
        application,
        base_url="http://127.0.0.1",
        headers={"x-studio-principal": "alice", "sec-fetch-site": "none"},
    ) as client:
        response = client.post("/api/candidates/" + route, json=body)
        assert response.status_code == 202, response.text
        status_route = response.json()["status_route"]
        completed = finish_candidate_job(client, status_route, timeout=90)
        assert completed["owner"] == "alice"
        assert completed["execution_model"] == "process"
        assert completed["result"]["candidate_sha256"] == response.json()["candidate_sha256"]
        assert client.get("/api/candidates/jobs").json()["jobs"][0]["job_id"] == completed["job_id"]
        assert client.get(status_route + "/artifacts/result.json").status_code == 200
        for suffix, method in [("", "GET"), ("/cancel", "POST"), ("/artifacts/result.json", "GET")]:
            assert (
                client.request(
                    method, status_route + suffix, headers={"x-studio-principal": "bob"}
                ).status_code
                == 404
            )
        assert client.post(status_route + "/cancel").json()["status"] == "completed"
        assert (
            client.get("/api/candidates/jobs", headers={"x-studio-principal": "bob"}).json()["jobs"]
            == []
        )
        # Check the authority itself, independently of the router's owned read.
        with (
            delegated(
                Principal("bob", frozenset()),
                method="POST",
                route="/api/candidates/jobs/{job_id}/cancel",
                request_id="deny-cancel",
            ),
            pytest.raises(KeyError),
        ):
            manager.cancel(completed["job_id"])
        with (
            delegated(
                Principal("bob", frozenset()),
                method="GET",
                route="/api/candidates/jobs/{job_id}/artifacts/{artifact_path:path}",
                request_id="deny-artifact",
            ),
            pytest.raises(KeyError),
        ):
            manager.read_artifact(completed["job_id"], "result.json")
    assert manager.generation_failures == {}
