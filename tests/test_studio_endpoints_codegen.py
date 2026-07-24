# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio endpoints codegen

"""Focused suite: TestCodegenEndpoint from former test_studio_endpoints.py."""

from __future__ import annotations

from tests.studio_endpoints_support import *  # noqa: F403


class TestCodegenEndpoint:
    def test_codegen_model(self, client):
        r = client.post(
            "/api/codegen",
            json={
                "mode": "model",
                "model_name": MODEL,
                "params": {},
                "dt": 0.1,
                "duration": 100,
                "current": 10,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "script" in data
        assert "oneliner" in data
        assert MODEL in data["script"]

    def test_codegen_ode(self, client):
        r = client.post(
            "/api/codegen",
            json={
                "mode": "ode",
                "equations": ["dv/dt = I"],
                "params": {},
                "init": {"v": 0},
                "dt": 0.1,
                "duration": 100,
                "current": 10,
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "script" in data
