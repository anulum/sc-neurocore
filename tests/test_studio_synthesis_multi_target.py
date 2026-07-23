# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio synthesis multi target

"""Focused suite: TestMultiTarget from former test_studio_synthesis.py."""

from __future__ import annotations

from tests.studio_synthesis_support import *  # noqa: F403

class TestMultiTarget:
    def test_multi_target_returns_all(self):
        verilog = "module test(); endmodule"
        result = multi_target_synthesis(verilog)
        assert "targets" in result
        assert "supported" in result
        assert set(result["supported"]) == set(_TARGETS.keys())
        for target in _TARGETS:
            assert target in result["targets"]

    def test_multi_target_endpoint(self, client):
        verilog = "module test(); endmodule"
        r = client.post("/api/synth/multi-target", json={"verilog": verilog})
        assert r.status_code == 200
        data = r.json()
        assert "targets" in data
        assert "supported" in data

    def test_multi_target_requires_verilog(self, client):
        r = client.post("/api/synth/multi-target", json={})
        assert r.status_code == 422

    def test_multi_target_rejects_non_string_verilog(self, client):
        r = client.post(
            "/api/synth/multi-target", json={"verilog": {"rtl": "module x();endmodule"}}
        )
        assert r.status_code == 422

    def test_multi_target_rejects_oversized_verilog(self, large_body_client):
        huge = "module x;\n" + ("wire a;\n" * 400_000) + "endmodule\n"
        r = large_body_client.post("/api/synth/multi-target", json={"verilog": huge})
        assert r.status_code == 422

