# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Candidate HTTP admission and actual process result custody

"""Exercise the application, registered workers and digest-bound artifact downloads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.api.candidate_jobs import build_candidate_jobs_router
from sc_neurocore.studio.api.candidates import build_candidates_router
from sc_neurocore.studio.api.runtime import build_studio_api_context
from sc_neurocore.studio.platform.jobs_manager import StudioJobManager
from sc_neurocore.studio.candidate_run import review_packet, simulate_candidate
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings
from tests.studio_candidate_jobs_support import finish_candidate_job
from tests.studio_candidate_support import adex_candidate


@pytest.mark.parametrize("operation", ["simulate", "review-packet"])
@pytest.mark.parametrize("numerics", ["euler", "gauss_seidel", "diffusion"])
def test_candidate_jobs_run_real_science_and_export_the_exact_admitted_inputs(
    operation: str, numerics: str, tmp_path: Path
) -> None:
    """Both HTTP entrypoints return202 and complete in real bounded process workers."""
    document = adex_candidate()
    document["model"]["state"] = {"w": 0.0, "v": -65.0}
    if numerics == "gauss_seidel":
        document["model"]["integration"]["method"] = "gauss_seidel"
        document["model"]["state"]["w"] = 7.0
        original = document["model"]["dynamics"]
        document["model"]["dynamics"] = {"w": original["w"], "v": original["v"]}
    elif numerics == "diffusion":
        for name, magnitude, unit in [
            ("sigma_v", 0.5, "mV / ms**0.5"),
            ("sigma_w", 0.25, "pA / ms**0.5"),
        ]:
            document["model"]["parameters"][name] = magnitude
            document["units"]["parameters"][name] = unit
            document["model"]["profile"]["parameters"][name] = "source"
            document["model"]["profile"]["units"][name] = unit
        document["model"]["dynamics"] = {"w": "sigma_w * xi", "v": "sigma_v * xi"}
        document["model"]["threshold"]["rng_seed"] = 17
    document["reference_tests"] = document["reference_tests"][:1]
    document["reference_tests"][0]["steps"] = 20
    body: dict[str, Any] = {"candidate": document}
    if operation == "simulate":
        body.update(current=800.0, steps=20)
    expected = (
        simulate_candidate(document, current=800.0, steps=20)
        if operation == "simulate"
        else review_packet(document)
    )
    if numerics == "gauss_seidel":
        # Sequential integration explicitly reads earlier updates in declared
        # equation order. A sorted JSON envelope must fail this control, while
        # the actual ordered seed must reproduce the original kernel exactly.
        sorted_document = json.loads(json.dumps(document, sort_keys=True))
        reordered = (
            simulate_candidate(sorted_document, current=800.0, steps=20)
            if operation == "simulate"
            else review_packet(sorted_document)
        )
        observations = "trace" if operation == "simulate" else "reference_tests"
        assert reordered[observations] != expected[observations]
        assert reordered["candidate_sha256"] == expected["candidate_sha256"]
    app = create_app(StudioRuntimeSettings(job_root_path=str(tmp_path / "jobs")))
    with TestClient(app, base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}) as client:
        response = client.post(f"/api/candidates/{operation}", json=body)
        assert response.status_code == 202, response.text
        receipt = response.json()
        record = finish_candidate_job(client, receipt["status_route"])
        assert record["execution_model"] == "process"
        assert record["owner"] == "local"
        assert record["result"] == expected
        assert record["experiment_sha256"] == receipt["request_sha256"]
        assert record["admission"]["projected_steps"] == 20
        assert {artifact["relative_path"] for artifact in record["artifacts"]} == {
            "candidate.json",
            "request.json",
            "result.json",
        }
        for artifact in record["artifacts"]:
            download = client.get(
                receipt["status_route"] + "/artifacts/" + artifact["relative_path"]
            )
            assert download.status_code == 200, download.text
            assert hashlib.sha256(download.content).hexdigest() == artifact["sha256"]
            assert download.headers["x-studio-artifact-sha256"] == artifact["sha256"]
            if artifact["relative_path"] == "request.json":
                assert hashlib.sha256(download.content).hexdigest() == receipt["request_sha256"]
                assert list(download.json()["candidate"]["model"]["state"]) == ["w", "v"]
            if artifact["relative_path"] == "result.json":
                assert json.loads(download.content) == expected
        history = client.get("/api/candidates/jobs").json()
        assert history["jobs"][0]["job_id"] == record["job_id"]
        assert history["jobs"][0]["result"] is None
        assert app.state.studio_job_manager.unreaped_workers == ()


def test_invalid_candidate_job_never_allocates_a_durable_record(tmp_path: Path) -> None:
    """Dimensional and simulation-bound failures refuse before ledger or worker custody."""
    app = create_app(StudioRuntimeSettings(job_root_path=str(tmp_path / "jobs")))
    document = adex_candidate()
    document["units"]["state"]["v"] = "ms"
    with TestClient(app, base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}) as client:
        before = client.get("/api/candidates/jobs").json()
        for operation in ["simulate", "review-packet"]:
            response = client.post(f"/api/candidates/{operation}", json={"candidate": document})
            assert response.status_code == 422
            assert response.json()["detail"]["reason"] == "invalid_candidate"
        assert (
            client.post(
                "/api/candidates/simulate", json={"candidate": adex_candidate(), "steps": 100_001}
            ).status_code
            == 422
        )
        for current in [True, "800"]:
            response = client.post(
                "/api/candidates/simulate",
                json={"candidate": adex_candidate(), "current": current, "steps": 2},
            )
            assert response.status_code == 422, response.text
        assert client.get("/api/candidates/jobs").json() == before


@pytest.mark.parametrize("refusal", ["bytes", "kind"])
def test_actual_admission_limits_refuse_without_starting_a_worker(
    refusal: str, tmp_path: Path
) -> None:
    """Configured artifact and manager admission limits remain effective through HTTP."""
    application = FastAPI()
    context = build_studio_api_context(
        application,
        StudioRuntimeSettings(
            job_root_path=str(tmp_path / "jobs"),
            job_max_artifact_bytes=1 if refusal == "bytes" else 1_000_000,
        ),
    )
    context.studio_job_manager = StudioJobManager(
        root=tmp_path / "bounded",
        allowed_kinds=frozenset({"compiler"}),
        default_timeout_seconds=30,
    )
    application.include_router(build_candidates_router(context))
    application.include_router(build_candidate_jobs_router(context))
    with TestClient(application) as client:
        response = client.post(
            "/api/candidates/simulate", json={"candidate": adex_candidate(), "steps": 2}
        )
        assert response.status_code == (422 if refusal == "bytes" else 503), response.text
        assert response.json()["detail"]["reason"] == (
            "invalid_candidate_job" if refusal == "bytes" else "candidate_job_rejected"
        )
        assert client.get("/api/candidates/jobs").json()["jobs"] == []
    assert context.studio_job_manager.unreaped_workers == ()
