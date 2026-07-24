# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio network canvas endpoints

"""Focused suite: TestEndpoints from former test_studio_network_canvas.py."""

from __future__ import annotations

from tests.studio_network_canvas_support import *  # noqa: F403


class TestEndpoints:
    def test_models_endpoint(self, client):
        r = client.get("/api/graph/models")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_models_endpoint_surfaces_discovery_failure(self, client, monkeypatch):
        import sc_neurocore.studio.api.design as design_routes

        def _boom():
            raise RuntimeError("catalog failed")

        monkeypatch.setattr(design_routes, "graph_available_models", _boom)
        r = client.get("/api/graph/models")
        assert r.status_code == 500
        assert r.json()["detail"] == "Internal error"

    def test_create_population_endpoint(self, client):
        r = client.post("/api/graph/population", json={"label": "Test", "count": 50})
        assert r.status_code == 200
        data = r.json()
        assert data["label"] == "Test"
        assert data["count"] == 50

    def test_validate_endpoint(self, client):
        r = client.post("/api/graph/validate", json={"populations": [], "projections": []})
        assert r.status_code == 200
        data = r.json()
        assert data["valid"] is False
        assert len(data["errors"]) > 0

    def test_simulate_endpoint(self, client):
        exc = create_population(count=30, neuron_type="excitatory")
        inh = create_population(count=10, neuron_type="inhibitory")
        proj = create_projection(exc["id"], inh["id"])
        graph = {"populations": [exc, inh], "projections": [proj], "duration": 30.0}
        r = client.post("/api/graph/simulate", json=graph)
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is True

    def test_export_nir_endpoint(self, client):
        pop = create_population()
        r = client.post("/api/graph/export-nir", json={"populations": [pop], "projections": []})
        assert r.status_code == 200
        data = r.json()
        assert data["format"] == "nir"

    def test_import_nir_endpoint(self, client):
        nir = {"nodes": {"a": {"type": "LIF", "count": 10}}, "edges": []}
        r = client.post("/api/graph/import-nir", json=nir)
        assert r.status_code == 200
        data = r.json()
        assert len(data["populations"]) == 1

    def test_export_nir_endpoint_rejects_malformed_graph(self, client):
        r = client.post("/api/graph/export-nir", json={"populations": [{"count": 10}]})
        assert r.status_code == 422
        assert r.json()["detail"] == "Invalid input"

    def test_import_nir_endpoint_rejects_malformed_edges(self, client):
        r = client.post("/api/graph/import-nir", json={"nodes": {"a": {}}, "edges": [{}]})
        assert r.status_code == 422
        assert r.json()["detail"] == "Invalid input"

    def test_project_load_endpoint_returns_loaded_state(self, client, monkeypatch):
        """The project adapter returns a successful load result unchanged."""
        import sc_neurocore.studio.api.design as design_routes

        monkeypatch.setattr(
            design_routes,
            "load_project",
            lambda name: {"name": name, "state": {"zoom": 2}},
        )

        response = client.get("/api/project/load/example")

        assert response.status_code == 200
        assert response.json() == {"name": "example", "state": {"zoom": 2}}

    def test_project_delete_endpoint_maps_missing_project(self, client, monkeypatch):
        """A missing project maps to the established not-found response."""
        import sc_neurocore.studio.api.design as design_routes

        monkeypatch.setattr(
            design_routes,
            "delete_project",
            lambda name: {"error": f"Project not found: {name}"},
        )

        response = client.delete("/api/project/missing")

        assert response.status_code == 404
        assert response.json()["detail"] == "Project not found: missing"

    def test_project_delete_endpoint_returns_deleted_project(self, client, monkeypatch):
        """A successful project deletion returns its public confirmation."""
        import sc_neurocore.studio.api.design as design_routes

        monkeypatch.setattr(
            design_routes,
            "delete_project",
            lambda name: {"deleted": name},
        )

        response = client.delete("/api/project/example")

        assert response.status_code == 200
        assert response.json() == {"deleted": "example"}

    def test_create_projection_endpoint_filters_unknown_fields(self, client, monkeypatch):
        """Only the graph projection contract reaches the implementation."""
        import sc_neurocore.studio.api.design as design_routes

        captured: dict[str, object] = {}

        def _create_projection(**kwargs):
            captured.update(kwargs)
            return {"id": "projection-1", **kwargs}

        monkeypatch.setattr(design_routes, "create_projection", _create_projection)
        response = client.post(
            "/api/graph/projection",
            json={"source_id": "a", "target_id": "b", "weight": 0.5, "private": "drop"},
        )

        assert response.status_code == 200
        assert captured == {"source_id": "a", "target_id": "b", "weight": 0.5}
        assert "private" not in response.json()
