# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio advanced analysis

"""Focused suite: TestAnalysisFunctions from former test_studio_advanced.py."""

from __future__ import annotations

from tests.studio_advanced_support import *  # noqa: F403


class TestAnalysisFunctions:
    def test_bifurcation_endpoint(self, client):
        r = client.post(
            "/api/bifurcation",
            json={
                "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
                "threshold": "v > -50",
                "reset": "v = -65",
                "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
                "init": {"v": -65.0},
                "dt": 0.1,
                "duration": 100,
                "current": 30,
                "sweep_param": "C",
                "sweep_min": 0.5,
                "sweep_max": 3.0,
                "sweep_steps": 5,
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert len(d["param_values"]) == 5
        assert len(d["attractors"]) == 5

    def test_sensitivity_endpoint(self, client):
        r = client.post(
            "/api/sensitivity",
            json={
                "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
                "threshold": "v > -50",
                "reset": "v = -65",
                "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
                "init": {"v": -65.0},
                "dt": 0.1,
                "duration": 100,
                "current": 30,
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert "sensitivities" in d
        assert len(d["sensitivities"]) > 0

    def test_precision_endpoint(self, client):
        r = client.post(
            "/api/precision",
            json={
                "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
                "threshold": "v > -50",
                "reset": "v = -65",
                "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
                "init": {"v": -65.0},
                "dt": 0.1,
                "duration": 50,
                "current": 30,
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert "error" in d
        assert d["error"]["max_error"] >= 0

    def test_heatmap_endpoint(self, client):
        r = client.post(
            "/api/heatmap",
            json={
                "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
                "threshold": "v > -50",
                "reset": "v = -65",
                "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
                "init": {"v": -65.0},
                "dt": 0.1,
                "duration": 50,
                "current": 30,
                "param_x": "tau_m",
                "x_min": 5,
                "x_max": 20,
                "x_steps": 3,
                "param_y": "C",
                "y_min": 0.5,
                "y_max": 2.0,
                "y_steps": 3,
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert len(d["rates"]) == 3
        assert len(d["rates"][0]) == 3

    def test_compile_endpoint(self, client):
        r = client.post(
            "/api/compile",
            json={
                "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
                "threshold": "v > -50",
                "reset": "v = -65",
                "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
            },
        )
        assert r.status_code == 200
        d = r.json()
        assert "module" in d["verilog"]
        assert d["chars"] > 100

    def test_codegen_endpoint(self, client):
        r = client.post(
            "/api/codegen",
            json={
                "mode": "ode",
                "equations": ["dv/dt = -(v + 65) / 10 + I"],
                "params": {},
                "init": {"v": -65},
                "dt": 0.1,
                "duration": 100,
                "current": 30,
            },
        )
        assert r.status_code == 200
        assert "from_equations" in r.json()["script"]
