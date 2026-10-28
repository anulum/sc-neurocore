# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio IR construction HTTP contracts

"""The actual build route lowers authored state and reports its native contract."""

from __future__ import annotations

from starlette.testclient import TestClient

from tests.studio_compiler_support import LIF_IR_REQUEST, client as client


class TestIRBuild:
    """Exercise the bounded native build endpoint with actual authored dynamics."""

    def test_ir_build_returns_ir_text(self, client: TestClient) -> None:
        """Return actual source-bound native graph text."""
        r = client.post("/api/ir/build", json=LIF_IR_REQUEST)
        assert r.status_code == 200
        data = r.json()
        assert "ir_text" in data
        assert len(data["ir_text"]) > 0
        assert "errors" in data
        assert isinstance(data["errors"], list)

    def test_ir_build_has_graph_metadata(self, client: TestClient) -> None:
        """Read the native graph name and actual interface counts."""
        r = client.post("/api/ir/build", json=LIF_IR_REQUEST)
        data = r.json()
        assert data["n_ops"] > 0
        assert data["n_inputs"] > 0
        assert data["n_outputs"] > 0
        assert data["graph_name"] == "sc_equation_neuron"

    def test_ir_build_reports_actual_profile_and_bound_source(self, client: TestClient) -> None:
        """Report physical port widths and source identity without assumed Q8.8 parameters."""
        r = client.post("/api/ir/build", json=LIF_IR_REQUEST)
        assert r.status_code == 200
        data = r.json()
        assert "params_q88" not in data
        assert data["source_present"] is True
        assert data["source_sha256"]
        assert data["clocks_per_timestep"] >= 1
        assert {port["name"]: port["width"] for port in data["output_widths"]} == {
            "v_out": 16,
            "spike_out": 1,
        }
