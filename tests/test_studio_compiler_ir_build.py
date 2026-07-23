# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio compiler ir build

"""Focused suite: TestIRBuild from former test_studio_compiler.py."""

from __future__ import annotations

from tests.studio_compiler_support import *  # noqa: F403

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

