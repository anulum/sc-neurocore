# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio FastAPI backend

from __future__ import annotations

import logging

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.templates import TEMPLATES


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app, base_url="http://127.0.0.1")


class TestHealth:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


class TestTemplatesEndpoints:
    def test_list_templates(self, client):
        r = client.get("/api/templates")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == 5
        names = {t["name"] for t in data}
        assert "lif" in names

    def test_get_template(self, client):
        r = client.get("/api/templates/lif")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == "lif"
        assert "equations" in data

    def test_get_nonexistent_template(self, client):
        r = client.get("/api/templates/nonexistent")
        assert r.status_code == 404

    @pytest.mark.parametrize("name", list(TEMPLATES.keys()))
    def test_each_template_accessible(self, client, name):
        r = client.get(f"/api/templates/{name}")
        assert r.status_code == 200

    def test_internal_error_is_logged(self, client, monkeypatch, caplog):
        import sc_neurocore.studio.app as app_mod

        def _boom():
            raise RuntimeError("catalog exploded")

        monkeypatch.setattr(app_mod, "list_models", _boom)
        with caplog.at_level(logging.ERROR):
            r = client.get("/api/models")
        assert r.status_code == 500
        assert r.json()["detail"] == "Internal error"
        assert "catalog exploded" in caplog.text


class TestSimulateEndpoint:
    def test_simulate_lif(self, client):
        t = TEMPLATES["lif"]
        r = client.post(
            "/api/simulate",
            json={
                "equations": t["equations"],
                "threshold": t["threshold"],
                "reset": t["reset"],
                "params": t["params"],
                "init": t["init"],
                "dt": t["dt"],
                "duration": t["duration"],
                "current": t["current"],
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert data["spike_count"] > 0
        assert "time" in data
        assert "states" in data

    def test_simulate_minimal(self, client):
        r = client.post(
            "/api/simulate",
            json={
                "equations": ["dv/dt = I"],
                "init": {"v": 0.0},
                "dt": 0.1,
                "duration": 10.0,
                "current": 1.0,
            },
        )
        assert r.status_code == 200

    def test_simulate_bad_equation(self, client):
        r = client.post(
            "/api/simulate",
            json={
                "equations": ["v = I"],
                "dt": 0.1,
                "duration": 10.0,
            },
        )
        assert r.status_code == 422

    def test_simulate_invalid_dt(self, client):
        r = client.post(
            "/api/simulate",
            json={
                "equations": ["dv/dt = I"],
                "dt": -1.0,
                "duration": 10.0,
            },
        )
        assert r.status_code == 422

    def test_simulate_returns_all_state_vars(self, client):
        t = TEMPLATES["izhikevich"]
        r = client.post(
            "/api/simulate",
            json={
                "equations": t["equations"],
                "threshold": t["threshold"],
                "reset": t["reset"],
                "params": t["params"],
                "init": t["init"],
                "dt": t["dt"],
                "duration": 50.0,
                "current": t["current"],
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "v" in data["states"]
        assert "u" in data["states"]

    def test_models_simulate_accepts_model_name_alias(self, client):
        # The Studio frontend and every other model endpoint send ``model_name``;
        # /api/models/simulate must accept it (not only the bare ``name``).
        by_alias = client.post(
            "/api/models/simulate",
            json={
                "model_name": "AdExNeuron",
                "current": 50.0,
                "duration": 50.0,
                "dt": 0.1,
            },
        )
        assert by_alias.status_code == 200, by_alias.text
        assert "states" in by_alias.json()

        by_name = client.post(
            "/api/models/simulate",
            json={"name": "AdExNeuron", "current": 50.0, "duration": 50.0, "dt": 0.1},
        )
        assert by_name.status_code == 200, by_name.text
