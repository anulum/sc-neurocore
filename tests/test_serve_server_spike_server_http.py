# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeServerHTTP from former test_serve_server.py

"""Focused suite: TestSpikeServerHTTP from former test_serve_server.py."""

from __future__ import annotations

from tests.serve_server_support import *  # noqa: F403

class TestSpikeServerHTTP:
    def test_health(self, running_server):
        result = _get("/health")
        assert result["status"] == "ok"

    def test_info_get(self, running_server):
        result = _get("/info")
        assert "timestep" in result
        assert result["type"] == "MockNetwork"

    def test_step_endpoint(self, running_server):
        result = _post("/step", {"inputs": {"input": [1.0, 2.0]}})
        assert "outputs" in result
        assert result["timestep"] >= 1

    def test_reset_endpoint(self, running_server):
        time.sleep(0.2)
        result = _post("/reset", {})
        assert result["status"] == "reset"
        assert result["timestep"] == 0

    def test_info_post(self, running_server):
        result = _post("/info", {})
        assert "timestep" in result

    def test_not_found_post(self, running_server):
        try:
            _post("/nonexistent", {})
            pytest.fail("Should have raised")
        except urllib.error.HTTPError as e:
            assert e.code == 404

    def test_not_found_get(self, running_server):
        try:
            _get("/nonexistent")
            pytest.fail("Should have raised")
        except urllib.error.HTTPError as e:
            assert e.code == 404

    def test_bad_json(self, running_server):
        req = urllib.request.Request(
            f"http://127.0.0.1:{_HTTP_PORT}/step",
            data=b"not json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            pytest.fail("Should have raised")
        except urllib.error.HTTPError as e:
            assert e.code == 400
