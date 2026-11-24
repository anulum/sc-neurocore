# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio integration endpoints

"""Focused suite: TestEndpoints from former test_studio_integration.py."""

from __future__ import annotations

import re
from tests.studio_integration_support import (
    TestClient,
    create_population,
    create_projection,
)
from tests.studio_integration_support import client as client


class TestEndpoints:
    """The Studio project and pipeline endpoints."""

    def test_save_endpoint(self, client: TestClient) -> None:
        """Saving a project returns a completed, hashed record without its path."""
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
        """A save without a name is rejected with 422."""
        r = client.post("/api/project/save", json={"state": {}})
        assert r.status_code == 422

    def test_save_rejects_non_object_state_endpoint(
        self,
        client: TestClient,
    ) -> None:
        """A save with a non-object state is rejected with 422."""
        r = client.post(
            "/api/project/save",
            json={"name": "bad_state_endpoint", "state": ["not", "an", "object"]},
        )
        assert r.status_code == 422

    def test_save_rejects_invalid_hdl_identifier_endpoint(
        self,
        client: TestClient,
    ) -> None:
        """A save with an invalid HDL identifier is rejected with 422."""
        r = client.post(
            "/api/project/save",
            json={"name": "bad_ident_endpoint", "state": {"module_name": "bad-name"}},
        )
        assert r.status_code == 422

    def test_list_endpoint(self, client: TestClient) -> None:
        """The project list is a list."""
        r = client.get("/api/project/list")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_load_nonexistent_endpoint(self, client: TestClient) -> None:
        """Loading an unknown project gives 404."""
        r = client.get("/api/project/load/nonexistent_xyz")
        assert r.status_code == 404

    def test_load_invalid_name_endpoint(self, client: TestClient) -> None:
        """Loading a project with an invalid name is rejected with 422."""
        r = client.get("/api/project/load/bad%5Cname")
        assert r.status_code == 422

    def test_delete_invalid_name_endpoint(self, client: TestClient) -> None:
        """Deleting a project with an invalid name is rejected with 422."""
        r = client.delete("/api/project/bad%5Cname")
        assert r.status_code == 422

    def test_pipeline_endpoint(self, client: TestClient) -> None:
        """The pipeline endpoint runs all five steps in Q16.16 with a bit-true RTL match."""
        exc = create_population(count=3, neuron_type="excitatory")
        inh = create_population(count=2, neuron_type="inhibitory")
        proj = create_projection(exc["id"], inh["id"])
        graph = {"populations": [exc, inh], "projections": [proj], "duration": 3.0}
        r = client.post(
            "/api/pipeline/run", json={"graph": graph, "target": "ice40", "q_format": "Q16.16"}
        )
        assert r.status_code == 200
        data = r.json()
        # The route runs the whole chain on the caller's own network.
        assert set(data["steps"]) == {"validate", "simulate", "lower", "cosimulate", "synthesise"}
        assert data["steps"]["lower"]["q_format"] == "Q16.16"
        assert data["steps"]["cosimulate"]["rtl_matches_bit_true_model"] is True
        assert data["trace"]["input_sha256"] == data["steps"]["lower"]["input_sha256"]

    def test_pipeline_endpoint_empty_graph(self, client: TestClient) -> None:
        """The pipeline endpoint reports failure for an empty graph."""
        r = client.post(
            "/api/pipeline/run",
            json={"graph": {"populations": [], "projections": []}, "target": "ice40"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
