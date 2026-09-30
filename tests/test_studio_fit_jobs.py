# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real background fitting, custody and replay

"""Exercise scientific fits through the application and real process workers."""

from __future__ import annotations

import time
from typing import Any

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from tests.test_studio_fits_routes import _body


def _finish(client: TestClient, route: str) -> dict[str, Any]:
    """Wait for the submitted production task and surface terminal errors."""
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        response = client.get(route)
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] == "completed":
            return dict(job)
        assert job["status"] not in {"failed", "timed_out", "cancelled"}, job
        time.sleep(0.05)
    raise AssertionError("background fit did not terminate")


def test_real_background_fit_and_replay() -> None:
    """A process fit exports full artifacts and replays its scientific digest."""
    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        response = client.post("/api/fits/jobs", json=_body(generations=2, population=4))
        assert response.status_code == 202, response.text
        job = _finish(client, response.json()["status_route"])
        assert job["execution_model"] == "process"
        assert job["owner"] == "local"
        assert {row["relative_path"] for row in job["artifacts"]} == {
            "experiment.json",
            "result.json",
            "history.jsonl",
        }
        replay = client.post("/api/fits/replay/jobs", json={"result": job["result"]})
        assert replay.status_code == 202, replay.text
        assert _finish(client, replay.json()["status_route"])["result"]["reproduced"] is True


def test_cohort_process_and_replay_use_the_full_admitted_document() -> None:
    """Background sweeps execute and export all exact shared samples and failures."""
    from sc_neurocore.fitting.cohort import cohort_sha256
    from tests.test_fitting_cohorts import cohort
    from tests.test_fitting_cohorts import receipts as receipts_for

    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        document = cohort().to_public_dict()
        response = client.post("/api/cohorts/jobs", json={"cohort": document})
        assert response.status_code == 202, response.text
        job = _finish(client, response.json()["status_route"])
        result = job["result"]
        assert result["cohort"] == document
        assert len(result["trials"]) == 4
        replay = client.post("/api/cohorts/replay", json={"result": result})
        assert replay.status_code == 200, replay.text
        assert replay.json()["reproduced"] is True
        report = client.post("/api/cohorts/measurements", json={"result": result, "receipts": []})
        assert report.status_code == 200
        assert report.json()["comparable"] is False
        receipts = receipts_for(result)
        del receipts[0]["latency_ms"]
        receipts[0]["receipt_sha256"] = cohort_sha256(
            {k: v for k, v in receipts[0].items() if k != "receipt_sha256"}
        )
        malformed = client.post(
            "/api/cohorts/measurements", json={"result": result, "receipts": receipts}
        )
        assert malformed.status_code == 200, malformed.text
        assert malformed.json()["reason"] == (
            "the cohort result or a measurement receipt is malformed"
        )
        assert "latency_ms" not in malformed.text
        compared = client.post(
            "/api/cohorts/measurements", json={"result": result, "receipts": receipts_for(result)}
        )
        assert compared.status_code == 200, compared.text
        assert compared.json()["comparable"] is True
        assert compared.json()["reason"] is None


def test_authenticated_users_cannot_observe_or_cancel_each_others_lab_jobs() -> None:
    """Real security middleware carries the authenticated actor into durable job custody."""
    from sc_neurocore.studio.platform.settings import StudioRuntimeSettings

    settings = StudioRuntimeSettings(enforce_route_policies=True)
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        alice = {"x-studio-principal": "alice"}
        bob = {"x-studio-principal": "bob"}
        anonymous = client.post("/api/fits/jobs", json=_body(generations=2, population=4))
        assert anonymous.status_code == 401
        response = client.post(
            "/api/fits/jobs", headers=alice, json=_body(generations=2, population=4)
        )
        assert response.status_code == 202, response.text
        route = response.json()["status_route"]
        assert client.get(route, headers=bob).status_code == 404
        assert client.post(route + "/cancel", headers=bob).status_code == 404
        assert client.get("/api/fits/jobs/nonexistent", headers=alice).status_code == 404
        assert client.get(route, headers=alice).json()["owner"] == "alice"
        cancelled = client.post(route + "/cancel", headers=alice)
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] in {"cancelling", "cancelled", "completed"}


def test_overbudget_invalid_and_malformed_lab_requests_are_refused_before_submission() -> None:
    """Bad protocol/version/split/optimiser documents never start an execution worker."""
    from tests.test_fitting_cohorts import cohort
    from sc_neurocore.fitting.problem import Recording, FitProblem, ParameterDomain
    from sc_neurocore.neurons.universal_dsl import load_schema

    app = create_app()
    with TestClient(app, base_url="http://127.0.0.1") as client:
        before = app.state.studio_job_manager.status().to_public_dict()
        assert client.post("/api/fits/jobs", json=_body(holdout=[])).status_code == 422
        assert (
            client.post("/api/cohorts/jobs", json={"cohort": {"schema_version": "bad"}}).status_code
            == 422
        )
        assert client.post("/api/cohorts/replay", json={"result": {}}).status_code == 422
        assert client.post("/api/fits/replay/jobs", json={"result": {}}).status_code == 422
        huge = _body(generations=500, population=100)
        huge["train"][0]["current"] = [0.0] * 3000
        huge["train"][0]["observed"] = [0.0] * 3000
        assert client.post("/api/fits/jobs", json=huge).status_code == 422
        problem = FitProblem(
            load_schema("lif"),
            "v",
            (ParameterDomain("R", 0.1, 2.0),),
            (Recording("train", (0.0,) * 3000, (0.0,) * 3000),),
            (Recording("holdout", (0.0, 0.0), (1.0, 1.0)),),
            0,
        )
        exported = {
            "problem": problem.to_public_dict(),
            "provenance": {"generations": 500, "population": 100},
        }
        # Two fitted parameters make the estimate exceed the job admission bound.
        exported["problem"]["domains"].append(
            {"name": "tau_m", "low": 1.0, "high": 20.0, "scale": "linear"}
        )
        assert client.post("/api/fits/replay/jobs", json={"result": exported}).status_code == 422
        assert client.post("/api/fits/replay", json={"result": exported}).status_code == 422
        invalid = cohort().to_public_dict()
        invalid["samples"][1]["group"] = invalid["samples"][0]["group"]
        assert client.post("/api/cohorts/jobs", json={"cohort": invalid}).status_code == 422
        after = app.state.studio_job_manager.status().to_public_dict()
        assert before == after


def test_document_byte_admission_and_worker_operation_refusal() -> None:
    """Real HTTP size refusal and the actual worker reject inadmissible requests."""
    from sc_neurocore.studio.platform.settings import StudioRuntimeSettings
    from tests.test_fitting_cohorts import cohort

    app = create_app(StudioRuntimeSettings(max_request_body_bytes=8_500_000))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        document = cohort().to_public_dict()
        document["noise_provenance"] = "x" * 4_100_000
        refused = client.post("/api/cohorts/jobs", json={"cohort": document})
        assert refused.status_code == 422
        assert "submission byte budget" in refused.json()["detail"]["message"]
        report = client.post(
            "/api/cohorts/measurements",
            json={"result": {"oversized": "x" * 4_100_000}, "receipts": []},
        )
        assert report.status_code == 422
        manager = app.state.studio_job_manager
        submitted = manager.submit_process_task(
            kind="analysis",
            owner="local",
            request_id=None,
            task_path="sc_neurocore.studio.api.fit_jobs:execute_laboratory_task",
            payload={"operation": "unsupported", "document": {}},
            admission={"laboratory_task": "fit"},
        )
        result = manager.wait(submitted.job_id, timeout_seconds=20.0)
        assert result.status == "failed"
        assert result.result is None


def test_malformed_lab_documents_are_refused_without_exception_text() -> None:
    """Every laboratory route answers a structural fault with 422 and one fixed sentence."""
    from sc_neurocore.studio.api.fits import MALFORMED_DOCUMENT
    from tests.test_fitting_cohorts import cohort

    document = cohort().to_public_dict()
    requests: list[tuple[str, dict[str, Any]]] = [
        ("/api/fits/replay/jobs", {"result": {}}),
        ("/api/fits/replay/jobs", {"result": {"problem": 5}}),
        ("/api/cohorts/jobs", {"cohort": {**document, "samples": 5}}),
        ("/api/cohorts/replay", {"result": {}}),
        ("/api/cohorts/replay", {"result": {"cohort": 5}}),
        ("/api/cohorts/replay", {"result": {"cohort": document}}),
    ]
    app = create_app()
    with TestClient(app, base_url="http://127.0.0.1") as client:
        before = app.state.studio_job_manager.status().to_public_dict()
        for route, body in requests:
            response = client.post(route, json=body)
            assert response.status_code == 422, (route, response.text)
            assert response.json()["detail"] == {
                "reason": "invalid_laboratory_request",
                "message": MALFORMED_DOCUMENT,
            }, route
            for leaked in ("'problem'", "'cohort'", "'trials'", "int", "attribute"):
                assert leaked not in response.text, (route, leaked)
        after = app.state.studio_job_manager.status().to_public_dict()
        assert after == before
