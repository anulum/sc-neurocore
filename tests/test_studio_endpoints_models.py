# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio endpoints models

"""Focused suite: TestModelEndpoints from former test_studio_endpoints.py."""

from __future__ import annotations

from tests.studio_endpoints_support import *  # noqa: F403


class TestModelEndpoints:
    def test_list_models(self, client):
        r = client.get("/api/models")
        assert r.status_code == 200
        data = r.json()
        assert len(data) > 100

    def test_model_detail(self, client):
        r = client.get(f"/api/models/{MODEL}")
        assert r.status_code == 200
        data = r.json()
        assert data["name"] == MODEL
        assert "params" in data
        assert "state_vars" in data
        assert "category" in data

    def test_model_detail_not_found(self, client):
        r = client.get("/api/models/NonexistentModel")
        assert r.status_code == 404

    def test_model_detail_internal_failure_is_500(self, client, monkeypatch):
        import sc_neurocore.studio.api.catalogue as catalogue_routes

        def _boom(_name: str):
            raise RuntimeError("metadata exploded")

        monkeypatch.setattr(catalogue_routes, "get_model_detail", _boom)
        r = client.get(f"/api/models/{MODEL}")
        assert r.status_code == 500
        assert r.json()["detail"] == "Internal error"

    def test_simulate_model(self, client):
        r = client.post(
            "/api/models/simulate",
            json={
                "name": MODEL,
                "duration": 50.0,
                "current": 10.0,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "time" in data
        assert "states" in data
