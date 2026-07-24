# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio compiler sv emit

"""Focused suite: TestSVEmit from former test_studio_compiler.py."""

from __future__ import annotations

from tests.studio_compiler_support import *  # noqa: F403


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
