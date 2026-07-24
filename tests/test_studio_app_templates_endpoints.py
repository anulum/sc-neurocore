# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio app templates endpoints

"""Focused suite: TestTemplatesEndpoints from former test_studio_app.py."""

from __future__ import annotations

from tests.studio_app_support import *  # noqa: F403


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
        import sc_neurocore.studio.api.catalogue as catalogue_routes

        def _boom():
            raise RuntimeError("catalog exploded")

        monkeypatch.setattr(catalogue_routes, "list_models", _boom)
        with caplog.at_level(logging.ERROR):
            r = client.get("/api/models")
        assert r.status_code == 500
        assert r.json()["detail"] == "Internal error"
        assert "catalog exploded" in caplog.text
