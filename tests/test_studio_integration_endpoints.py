# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio integration endpoints

"""Focused suite: TestEndpoints from former test_studio_integration.py."""

from __future__ import annotations

from tests.studio_integration_support import *  # noqa: F403


class TestEndpoints:
    def test_save_endpoint(self, client: TestClient) -> None:
        r = client.post(
            "/api/project/save",
            json={"name": "endpoint_test", "state": {"dt": 0.1}},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "endpoint_test"
        assert data["schema_version"] == "studio.project-save.v1"
        assert data["evidence_classification"] == "project_workspace"
        assert data["status"] == "completed"
        assert re.fullmatch(r"[0-9a-f]{64}", data["state_sha256"])
        assert re.fullmatch(r"[0-9a-f]{64}", data["project_sha256"])
        assert "path" not in data

    def test_save_requires_name(self, client: TestClient) -> None:
        r = client.post("/api/project/save", json={"state": {}})
        assert r.status_code == 422

    def test_save_rejects_non_object_state_endpoint(
        self,
        client: TestClient,
    ) -> None:
        r = client.post(
            "/api/project/save",
            json={"name": "bad_state_endpoint", "state": ["not", "an", "object"]},
        )
        assert r.status_code == 422

    def test_save_rejects_invalid_hdl_identifier_endpoint(
        self,
        client: TestClient,
    ) -> None:
        r = client.post(
            "/api/project/save",
            json={"name": "bad_ident_endpoint", "state": {"module_name": "bad-name"}},
        )
        assert r.status_code == 422

    def test_list_endpoint(self, client: TestClient) -> None:
        r = client.get("/api/project/list")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_load_nonexistent_endpoint(self, client: TestClient) -> None:
        r = client.get("/api/project/load/nonexistent_xyz")
        assert r.status_code == 404

    def test_load_invalid_name_endpoint(self, client: TestClient) -> None:
        r = client.get("/api/project/load/bad%5Cname")
        assert r.status_code == 422

    def test_delete_invalid_name_endpoint(self, client: TestClient) -> None:
        r = client.delete("/api/project/bad%5Cname")
        assert r.status_code == 422

    def test_pipeline_endpoint(self, client: TestClient) -> None:
        exc = create_population(count=20, neuron_type="excitatory")
        inh = create_population(count=5, neuron_type="inhibitory")
        proj = create_projection(exc["id"], inh["id"])
        graph = {"populations": [exc, inh], "projections": [proj], "duration": 20.0}
        r = client.post("/api/pipeline/run", json={"graph": graph, "target": "ice40"})
        assert r.status_code == 200
        data = r.json()
        assert "steps" in data

    def test_pipeline_endpoint_empty_graph(self, client: TestClient) -> None:
        r = client.post(
            "/api/pipeline/run",
            json={"graph": {"populations": [], "projections": []}, "target": "ice40"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
