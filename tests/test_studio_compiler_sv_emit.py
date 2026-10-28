# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio native and direct SystemVerilog emission HTTP contracts

"""Native IR emission and direct Python equation emission retain distinct contracts."""

from __future__ import annotations

from starlette.testclient import TestClient

from tests.studio_compiler_support import LIF_EQ, LIF_IR_REQUEST, client as client


class TestSVEmit:
    """Exercise both maintained SystemVerilog delivery paths through HTTP."""

    def test_emit_sv_from_ir(self, client: TestClient) -> None:
        """Replay bound source before emitting its native SystemVerilog."""
        build = client.post("/api/ir/build", json=LIF_IR_REQUEST).json()
        r = client.post("/api/ir/emit-sv", json={"ir_text": build["ir_text"]})
        assert r.status_code == 200
        data = r.json()
        assert "systemverilog" in data
        assert "module" in data["systemverilog"]
        assert len(data["systemverilog"]) > 50
        assert data["source_replay"] == "passed"

    def test_emit_sv_direct(self, client: TestClient) -> None:
        """Preserve the separate direct Python equation-to-Verilog contract."""
        request = {k: v for k, v in LIF_EQ.items() if k not in {"dt", "duration", "current"}}
        r = client.post("/api/ir/emit-sv-direct", json=request)
        assert r.status_code == 200
        data = r.json()
        assert "verilog" in data
        assert "module" in data["verilog"]
        assert data["module_name"] == "sc_ode_neuron"
