# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Graph route admission and identifier refusals

"""Exercise real closed providers and malformed identifiers through graph HTTP."""

from pathlib import Path

import pytest

from fastapi import FastAPI
from starlette.testclient import TestClient

from sc_neurocore.studio.api.graph_jobs import build_graph_jobs_router
from sc_neurocore.studio.api.runtime import build_studio_api_context
from sc_neurocore.studio.api.security import install_studio_security_middleware
from sc_neurocore.studio.platform.jobs import StudioJobManager
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings
from tests.test_studio_graph_jobs import client as client, graph


def test_graph_identifiers_are_closed_on_every_job_route(client: TestClient) -> None:
    """Malformed identifiers never reach a ledger or an artifact read."""
    for suffix, method in [("", "GET"), ("/cancel", "POST"), ("/result", "GET")]:
        response = client.request(method, "/api/graph/jobs/not-a-job" + suffix)
        assert response.status_code == 404
        assert response.json()["detail"] == "graph_job_not_found"
    assert client.get("/api/graph/jobs").json()["jobs"] == []


def test_operator_disallowed_analysis_refuses_before_a_graph_job_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production router honours a real provider's explicitly closed kind set."""
    monkeypatch.setenv("SC_NEUROCORE_STUDIO_PROJECT_ROOT", str(tmp_path / "projects"))
    app = FastAPI()
    context = build_studio_api_context(
        app,
        StudioRuntimeSettings(
            job_root_path=str(tmp_path / "unused-context-jobs"),
            audit_log_path=str(tmp_path / "audit.jsonl"),
            benchmark_databank_path=str(tmp_path / "databank.jsonl"),
        ),
    )
    manager = StudioJobManager(
        root=tmp_path / "restricted-jobs",
        allowed_kinds=frozenset({"compiler"}),
        default_timeout_seconds=30.0,
    )
    context.studio_job_manager = manager
    app.state.studio_job_manager = manager
    install_studio_security_middleware(app, context)
    app.include_router(build_graph_jobs_router(context))
    with TestClient(
        app, base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}
    ) as session:
        response = session.post("/api/graph/jobs", json=graph())
        assert response.status_code == 503
        assert response.json()["detail"] == "graph_job_admission_failed"
        assert session.get("/api/graph/jobs").json()["jobs"] == []
    assert manager.status().active_count == 0
