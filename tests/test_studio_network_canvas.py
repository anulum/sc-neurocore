# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio Network Canvas (Block 5)

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.network_graph import (
    available_models,
    create_population,
    create_projection,
    graph_to_nir,
    nir_to_graph,
    simulate_graph,
    validate_graph,
)


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(), base_url="http://127.0.0.1")


# --- Population & Projection Creation ---


class TestCreation:
    def test_create_population_defaults(self):
        pop = create_population()
        assert pop["type"] == "population"
        assert pop["id"].startswith("pop_")
        assert pop["count"] == 80
        assert pop["neuron_type"] == "excitatory"

    def test_create_population_inhibitory(self):
        pop = create_population(
            label="Inh 0", model="LIFNeuron", count=20, neuron_type="inhibitory"
        )
        assert pop["neuron_type"] == "inhibitory"
        assert pop["count"] == 20
        assert pop["label"] == "Inh 0"

    def test_create_projection(self):
        proj = create_projection("src", "tgt", weight=0.3, delay=2.0, probability=0.5)
        assert proj["id"].startswith("proj_")
        assert proj["source"] == "src"
        assert proj["target"] == "tgt"
        assert proj["weight"] == 0.3
        assert proj["delay"] == 2.0

    def test_unique_ids(self):
        ids = {create_population()["id"] for _ in range(10)}
        assert len(ids) == 10

    def test_available_models(self):
        models = available_models()
        assert isinstance(models, list)
        assert len(models) > 100
        assert "AdExNeuron" in models

    def test_available_models_raises_on_catalog_failure(self, monkeypatch):
        import sc_neurocore.studio.network_graph as mod

        def _boom():
            raise RuntimeError("catalog failed")

        monkeypatch.setattr(mod, "list_models", _boom)
        with pytest.raises(RuntimeError, match="catalog failed"):
            mod.available_models()


# --- Graph Validation ---


class TestValidation:
    def _make_graph(self, n_exc=80, n_inh=20):
        exc = create_population(count=n_exc, neuron_type="excitatory")
        inh = create_population(count=n_inh, neuron_type="inhibitory")
        proj = create_projection(exc["id"], inh["id"])
        return {"populations": [exc, inh], "projections": [proj]}

    def test_valid_graph(self):
        errors = validate_graph(self._make_graph())
        assert errors == []

    def test_empty_graph(self):
        errors = validate_graph({"populations": [], "projections": []})
        assert len(errors) > 0
        assert "no populations" in errors[0].lower()

    def test_dangling_projection(self):
        pop = create_population()
        proj = create_projection(pop["id"], "nonexistent")
        errors = validate_graph({"populations": [pop], "projections": [proj]})
        assert any("not found" in e for e in errors)

    def test_zero_weight_warning(self):
        graph = self._make_graph()
        graph["projections"][0]["weight"] = 0
        errors = validate_graph(graph)
        assert any("zero weight" in e for e in errors)

    def test_neuron_count_limit(self):
        graph = self._make_graph(n_exc=1500, n_inh=600)
        errors = validate_graph(graph)
        assert any("2000" in e for e in errors)

    def test_probability_out_of_range(self):
        graph = self._make_graph()
        graph["projections"][0]["probability"] = 1.5
        errors = validate_graph(graph)
        assert any("probability" in e.lower() for e in errors)


# --- Graph Simulation ---


class TestGraphSimulation:
    def test_simulate_valid_graph(self):
        exc = create_population(count=40, neuron_type="excitatory")
        inh = create_population(count=10, neuron_type="inhibitory")
        proj = create_projection(exc["id"], inh["id"])
        graph = {"populations": [exc, inh], "projections": [proj], "duration": 50.0, "dt": 0.1}
        result = simulate_graph(graph)
        assert result["success"] is True
        assert result["n_total"] == 50
        assert "graph_summary" in result

    def test_simulate_empty_graph(self):
        result = simulate_graph({"populations": [], "projections": []})
        assert result["success"] is False
        assert "errors" in result

    def test_simulate_rejects_unsupported_topology(self):
        exc0 = create_population(count=20, neuron_type="excitatory")
        exc1 = create_population(count=20, neuron_type="excitatory")
        inh = create_population(count=10, neuron_type="inhibitory")
        graph = {
            "populations": [exc0, exc1, inh],
            "projections": [create_projection(exc0["id"], inh["id"])],
            "duration": 10.0,
            "dt": 0.1,
        }

        result = simulate_graph(graph)

        assert result["success"] is False
        assert any("exactly one excitatory and one inhibitory" in e for e in result["errors"])


# --- NIR Export/Import ---


class TestNIR:
    def test_export_nir(self):
        exc = create_population(label="E", count=80)
        inh = create_population(label="I", count=20, neuron_type="inhibitory")
        proj = create_projection(exc["id"], inh["id"])
        graph = {"populations": [exc, inh], "projections": [proj]}
        nir = graph_to_nir(graph)
        assert nir["format"] == "nir"
        assert len(nir["nodes"]) == 2
        assert len(nir["edges"]) == 1

    def test_import_nir(self):
        nir = {
            "format": "nir",
            "version": "0.1",
            "nodes": {
                "pop_a": {"type": "LIF", "count": 80, "neuron_type": "excitatory"},
                "pop_b": {"type": "LIF", "count": 20, "neuron_type": "inhibitory"},
            },
            "edges": [{"source": "pop_a", "target": "pop_b", "weight": 0.5}],
        }
        graph = nir_to_graph(nir)
        assert len(graph["populations"]) == 2
        assert len(graph["projections"]) == 1
        assert graph["populations"][0]["id"] == "pop_a"

    def test_roundtrip(self):
        exc = create_population(label="E", count=64)
        inh = create_population(label="I", count=16, neuron_type="inhibitory")
        proj = create_projection(exc["id"], inh["id"], weight=0.3)
        graph = {"populations": [exc, inh], "projections": [proj]}
        nir = graph_to_nir(graph)
        restored = nir_to_graph(nir)
        assert len(restored["populations"]) == 2
        assert len(restored["projections"]) == 1


# --- Endpoints ---


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
