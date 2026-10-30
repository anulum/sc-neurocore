# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Candidate process lifecycle and durable recovery

"""Observe cancellation, deadlines, restart recovery and actual artifact custody."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings
from tests.studio_candidate_jobs_support import finish_candidate_job
from tests.studio_candidate_support import adex_candidate


_RESTART_CLIENT = """
import json, os, sys
from pathlib import Path
from starlette.testclient import TestClient
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings
from tests.studio_candidate_jobs_support import finish_candidate_job

root, mode = Path(sys.argv[1]), sys.argv[2]
settings = StudioRuntimeSettings(job_root_path=str(root / 'jobs'), enforce_route_policies=True)
headers = {'sec-fetch-site': 'none', 'x-studio-principal': 'alice'}
app = create_app(settings)
with TestClient(app, base_url='http://127.0.0.1', headers=headers) as client:
    if mode == 'write':
        document = json.loads((root / 'candidate.json').read_text())
        records = []
        artifacts = {}
        for operation in ['simulate', 'review-packet']:
            body = {'candidate': document}
            if operation == 'simulate':
                body.update(current=800.0, steps=3)
            response = client.post('/api/candidates/' + operation, json=body)
            assert response.status_code == 202, response.text
            receipt = response.json()
            record = finish_candidate_job(client, receipt['status_route'])
            records.append(record)
            artifacts[record['job_id']] = {}
            for item in record['artifacts']:
                name = item['relative_path']
                downloaded = client.get(receipt['status_route'] + '/artifacts/' + name)
                assert downloaded.status_code == 200, downloaded.text
                artifacts[record['job_id']][name] = downloaded.content.decode('utf-8')
        observed = {'pid': os.getpid(), 'records': records, 'artifacts': artifacts}
        (root / 'before-restart.json').write_text(json.dumps(observed))
    else:
        observed = json.loads((root / 'before-restart.json').read_text())
        assert observed['pid'] != os.getpid()
        for record in observed['records']:
            route = '/api/candidates/jobs/' + record['job_id']
            assert client.get(route).json() == record
            for name, expected in observed['artifacts'][record['job_id']].items():
                downloaded = client.get(route + '/artifacts/' + name)
                assert downloaded.status_code == 200, downloaded.text
                assert downloaded.content == expected.encode('utf-8')
            assert client.get(route, headers={'x-studio-principal': 'bob'}).status_code == 404
        assert {job['job_id'] for job in client.get('/api/candidates/jobs').json()['jobs']} == {
            record['job_id'] for record in observed['records']
        }
        assert all(record['status'] == 'completed' for record in observed['records'])
        assert {record['admission']['candidate_operation'] for record in observed['records']} == {
            'simulate', 'review'
        }
    assert app.state.studio_job_manager.unreaped_workers == ()
print(json.dumps({'pid': os.getpid(), 'mode': mode, 'job_count': len(observed['records'])}))
"""


def test_server_process_exit_preserves_both_candidate_results_without_resubmission(
    tmp_path: Path,
) -> None:
    """A distinct interpreter recovers both HTTP jobs and every original artifact byte."""
    document = adex_candidate()
    document["reference_tests"] = document["reference_tests"][:1]
    document["reference_tests"][0]["steps"] = 3
    (tmp_path / "candidate.json").write_text(json.dumps(document))
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    outcomes = []
    for mode in ["write", "read"]:
        completed = subprocess.run(
            [sys.executable, "-c", _RESTART_CLIENT, str(tmp_path), mode],
            env=environment,
            text=True,
            capture_output=True,
            timeout=90,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        outcomes.append(json.loads(completed.stdout))
    assert outcomes[0]["pid"] != outcomes[1]["pid"]
    assert [outcome["job_count"] for outcome in outcomes] == [2, 2]


@pytest.mark.parametrize("control", ["cancel", "deadline"])
def test_candidate_supervisor_stops_and_reaps_real_long_work(control: str, tmp_path: Path) -> None:
    """A requested stop and a finite deadline retain their distinct terminal verdicts."""
    settings = StudioRuntimeSettings(
        job_root_path=str(tmp_path / "jobs"),
        job_default_timeout_seconds=0.05 if control == "deadline" else 60,
    )
    app = create_app(settings)
    with TestClient(app, base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}) as client:
        submitted = client.post(
            "/api/candidates/simulate", json={"candidate": adex_candidate(), "steps": 100_000}
        )
        assert submitted.status_code == 202, submitted.text
        route = submitted.json()["status_route"]
        if control == "cancel":
            stopping = client.post(route + "/cancel")
            assert stopping.status_code == 200, stopping.text
            assert stopping.json()["status"] in {"cancelling", "cancelled"}
        deadline = time.monotonic() + 30
        while True:
            record = client.get(route).json()
            if record["status"] not in {"pending", "running", "cancelling"}:
                break
            assert time.monotonic() < deadline, record
            time.sleep(0.05)
        assert record["status"] == ("cancelled" if control == "cancel" else "timed_out")
        assert record["result"] is None
        assert app.state.studio_job_manager.unreaped_workers == ()


def test_application_restart_recovers_completed_result_and_detects_changed_artifact(
    tmp_path: Path,
) -> None:
    """A new application reads retained bytes and refuses a substituted artifact."""
    root = tmp_path / "jobs"
    settings = StudioRuntimeSettings(job_root_path=str(root), enforce_route_policies=True)
    headers = {"sec-fetch-site": "none", "x-studio-principal": "alice"}
    with TestClient(create_app(settings), base_url="http://127.0.0.1", headers=headers) as first:
        receipt = first.post(
            "/api/candidates/simulate", json={"candidate": adex_candidate(), "steps": 2}
        )
        assert receipt.status_code == 202, receipt.text
        route = receipt.json()["status_route"]
        completed = finish_candidate_job(first, route)
    with TestClient(create_app(settings), base_url="http://127.0.0.1", headers=headers) as second:
        assert second.get(route).json() == completed
        assert second.get("/api/candidates/jobs").json()["jobs"][0]["job_id"] == completed["job_id"]
        for suffix, method in [("", "GET"), ("/cancel", "POST"), ("/artifacts/result.json", "GET")]:
            assert (
                second.request(
                    method, route + suffix, headers={"x-studio-principal": "bob"}
                ).status_code
                == 404
            )
        assert (
            second.get("/api/candidates/jobs", headers={"x-studio-principal": "bob"}).json()["jobs"]
            == []
        )
        assert second.get(route + "/artifacts/missing.json").status_code == 404
        assert second.get("/api/candidates/jobs/not-a-job").status_code == 404
        assert second.get("/api/candidates/jobs/sj_0000000000000000").status_code == 404
        assert second.get("/api/candidates/jobs?limit=0").status_code == 422
        assert second.get("/api/candidates/jobs?limit=101").status_code == 422
        artifact = root / str(completed["job_id"]) / "result.json"
        assert artifact.is_file()
        artifact.write_text('{"substituted":true}')
        assert second.get(route + "/artifacts/result.json").status_code == 409
