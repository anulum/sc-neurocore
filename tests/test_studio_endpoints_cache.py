# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio endpoints cache

"""Focused suite: TestCacheStats from former test_studio_endpoints.py."""

from __future__ import annotations

from tests.studio_endpoints_support import *  # noqa: F403


class TestCacheStats:
    def test_cache_stats(self, client):
        r = client.get("/api/cache/stats")
        assert r.status_code == 200
        data = r.json()
        assert "hits" in data
        assert "misses" in data
        assert "size" in data

    def test_simulation_cache_reuses_and_evicts_results(self, monkeypatch):
        """The bounded simulation cache reuses hits and evicts its oldest entry."""
        import sc_neurocore.studio.api.simulation as simulation_routes

        cache = simulation_routes._SimCache(maxsize=1)
        monkeypatch.setattr(simulation_routes, "_cache", cache)
        local_client = TestClient(create_app(), base_url="http://127.0.0.1")
        payload = {
            "current": 1.0,
            "dt": 0.1,
            "duration": 1.0,
            "equations": ["dv/dt = I"],
            "init": {"v": 0.0},
        }

        first = local_client.post("/api/simulate", json=payload)
        cached = local_client.post("/api/simulate", json=payload)
        second = local_client.post("/api/simulate", json={**payload, "current": 2.0})
        evicted = local_client.post("/api/simulate", json=payload)

        assert [response.status_code for response in (first, cached, second, evicted)] == [
            200,
            200,
            200,
            200,
        ]
        assert cached.json() == first.json()
        assert cache.hits == 1
        assert cache.misses == 3
        assert len(cache._cache) == 1
