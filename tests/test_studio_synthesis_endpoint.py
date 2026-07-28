# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio synthesis endpoint

"""Focused suite: TestSynthesisEndpoint from former test_studio_synthesis.py."""

from __future__ import annotations

from tests.studio_synthesis_support import *  # noqa: F403


class TestSynthesisEndpoint:
    def test_synth_requires_verilog(self, client):
        r = client.post("/api/synth/run", json={"target": "ice40"})
        assert r.status_code == 422

    def test_synth_empty_verilog_rejected(self, client):
        r = client.post("/api/synth/run", json={"verilog": "", "target": "ice40"})
        assert r.status_code == 422

    def test_synth_non_string_verilog_rejected(self, client):
        r = client.post("/api/synth/run", json={"verilog": {"module": "x"}, "target": "ice40"})
        assert r.status_code == 422

    def test_synth_oversized_verilog_rejected(self, large_body_client):
        huge = "module x;\n" + ("wire a;\n" * 400_000) + "endmodule\n"
        r = large_body_client.post("/api/synth/run", json={"verilog": huge, "target": "ice40"})
        assert r.status_code == 422

    def test_synth_with_stub_verilog(self, client):
        verilog = "module test(); endmodule"
        r = client.post("/api/synth/run", json={"verilog": verilog, "target": "ice40"})
        assert r.status_code == 200
        data = r.json()
        assert "success" in data
        assert "target" in data
        assert data["target"] == "ice40"

    def test_synth_invalid_target(self, client):
        verilog = "module test(); endmodule"
        r = client.post("/api/synth/run", json={"verilog": verilog, "target": "invalid"})
        assert r.status_code == 422

    @pytest.mark.parametrize("missing", ["compile_traceability", "cosim_parity"])
    def test_terminal_requires_selected_rtl_evidence(self, client, missing):
        payload = {
            "compile_traceability": {},
            "cosim_parity": {},
            "target": "ecp5",
            "verilog": "module test(); endmodule",
        }
        payload.pop(missing)

        response = client.post("/api/synth/terminal", json=payload)

        assert response.status_code == 422

    def test_terminal_rejects_target_without_pnr(self, client):
        response = client.post(
            "/api/synth/terminal",
            json={
                "compile_traceability": {},
                "cosim_parity": {},
                "target": "gowin",
                "verilog": "module test(); endmodule",
            },
        )

        assert response.status_code == 422

    def test_synth_all_valid_targets(self, client):
        verilog = "module test(); endmodule"
        for target in _TARGETS:
            r = client.post("/api/synth/run", json={"verilog": verilog, "target": target})
            assert r.status_code == 200
            data = r.json()
            assert data["target"] == target
