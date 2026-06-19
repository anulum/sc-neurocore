# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio Compiler Inspector (Block 2)

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app

LIF_EQ = {
    "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
    "threshold": "v > -50",
    "reset": "v = -65",
    "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
    "init": {"v": -65.0},
    "dt": 0.1,
    "duration": 50.0,
    "current": 30.0,
}


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(), base_url="http://127.0.0.1")


class TestIRBuild:
    def test_ir_build_returns_ir_text(self, client):
        r = client.post("/api/ir/build", json=LIF_EQ)
        assert r.status_code == 200
        data = r.json()
        assert "ir_text" in data
        assert len(data["ir_text"]) > 0
        assert "errors" in data
        assert isinstance(data["errors"], list)

    def test_ir_build_has_graph_metadata(self, client):
        r = client.post("/api/ir/build", json=LIF_EQ)
        data = r.json()
        assert data["n_ops"] > 0
        assert data["n_inputs"] > 0
        assert data["n_outputs"] > 0
        assert data["graph_name"] == "ode_neuron"

    def test_ir_build_has_q88_params(self, client):
        r = client.post("/api/ir/build", json=LIF_EQ)
        data = r.json()
        assert "params_q88" in data
        assert "E_L" in data["params_q88"]


class TestIRVerify:
    def test_verify_valid_ir(self, client):
        build = client.post("/api/ir/build", json=LIF_EQ).json()
        r = client.post("/api/ir/verify", json={"ir_text": build["ir_text"]})
        assert r.status_code == 200
        data = r.json()
        assert data["valid"] is True
        assert data["errors"] == []

    def test_verify_empty_ir_fails(self, client):
        r = client.post("/api/ir/verify", json={"ir_text": ""})
        assert r.status_code == 422


class TestSVEmit:
    def test_emit_sv_from_ir(self, client):
        build = client.post("/api/ir/build", json=LIF_EQ).json()
        r = client.post("/api/ir/emit-sv", json={"ir_text": build["ir_text"]})
        assert r.status_code == 200
        data = r.json()
        assert "systemverilog" in data
        assert "module" in data["systemverilog"]
        assert data["chars"] > 50

    def test_emit_sv_direct(self, client):
        r = client.post("/api/ir/emit-sv-direct", json=LIF_EQ)
        assert r.status_code == 200
        data = r.json()
        assert "verilog" in data
        assert "module" in data["verilog"]
        assert data["module_name"] == "sc_ode_neuron"


class TestCosim:
    def test_cosim_returns_traces(self, client):
        r = client.post("/api/ir/cosim", json=LIF_EQ)
        assert r.status_code == 200
        data = r.json()
        assert "float_result" in data
        assert "fixed_result" in data
        assert "error" in data
        assert data["error"]["max_error"] >= 0
