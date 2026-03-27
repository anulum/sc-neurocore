# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio Synthesis Dashboard (Block 3)

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.synthesis import check_tools


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


class TestToolDetection:
    def test_check_tools_returns_dict(self):
        result = check_tools()
        assert "yosys" in result
        assert "nextpnr_ice40" in result
        for tool_info in result.values():
            assert "available" in tool_info
            assert "version" in tool_info

    def test_tools_status_endpoint(self, client):
        r = client.get("/api/synth/tools-status")
        assert r.status_code == 200
        data = r.json()
        assert "yosys" in data


class TestSynthesisEndpoint:
    def test_synth_requires_verilog(self, client):
        r = client.post("/api/synth/run", json={"target": "ice40"})
        assert r.status_code == 422

    def test_synth_with_stub_verilog(self, client):
        verilog = "module test(); endmodule"
        r = client.post("/api/synth/run", json={"verilog": verilog, "target": "ice40"})
        assert r.status_code == 200
        data = r.json()
        assert "success" in data
        assert "target" in data

    def test_synth_invalid_target(self, client):
        verilog = "module test(); endmodule"
        r = client.post("/api/synth/run", json={"verilog": verilog, "target": "invalid"})
        assert r.status_code == 422


class TestPnREndpoint:
    def test_pnr_requires_json_path(self, client):
        r = client.post("/api/synth/pnr", json={"target": "ice40"})
        assert r.status_code == 422
