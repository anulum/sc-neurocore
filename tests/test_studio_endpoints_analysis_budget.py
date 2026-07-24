# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio endpoints analysis budget

"""Focused suite: TestAnalysisBudgetEnforcement from former test_studio_endpoints.py."""

from __future__ import annotations

from tests.studio_endpoints_support import *  # noqa: F403


class TestAnalysisBudgetEnforcement:
    def test_heatmap_rejected_over_step_budget(self) -> None:
        client = _budget_client(
            max_sync_analysis_steps_per_simulation=1_000,
            max_sync_analysis_simulations=1_000,
        )
        r = client.post(
            "/api/heatmap",
            json={
                "model_name": MODEL,
                "duration": 200.0,
                "dt": 0.001,
                "param_x": "v_rest",
                "x_min": -75,
                "x_max": -55,
                "x_steps": 3,
                "param_y": "a",
                "y_min": 0,
                "y_max": 5,
                "y_steps": 3,
            },
        )
        assert r.status_code == 422
        assert r.json()["detail"]["limit"] == "steps_per_simulation"
        assert r.json()["detail"]["allowed"] == 1_000
        assert r.json()["detail"]["projected"] == 200_000

    def test_sensitivity_rejected_over_simulation_budget(self) -> None:
        client = _budget_client(max_sync_analysis_simulations=5)
        params = {f"p{i}": float(i + 1) for i in range(10)}
        r = client.post(
            "/api/sensitivity",
            json={"model_name": MODEL, "duration": 20.0, "params": params},
        )
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert detail["limit"] == "simulations"
        assert detail["projected"] == 1 + 2 * 10
        assert detail["allowed"] == 5

    def test_multi_simulate_rejected_over_simulation_budget(self) -> None:
        client = _budget_client(max_sync_analysis_simulations=1)
        r = client.post(
            "/api/multi-simulate",
            json=[
                {"name": MODEL, "duration": 20.0, "current": 10.0},
                {"name": "ChayNeuron", "duration": 20.0, "current": 10.0},
            ],
        )
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert detail["limit"] == "simulations"
        assert detail["projected"] == 2
        assert detail["allowed"] == 1

    def test_compare_invalid_cost_fields_reaches_payload_validation(
        self,
        client: TestClient,
    ) -> None:
        invalid_config = {
            "equations": ["dv/dt = I"],
            "init": {"v": 0.0},
            "duration": "not-a-number",
            "dt": "not-a-number",
        }
        r = client.post(
            "/api/compare",
            json={"config_a": invalid_config, "config_b": invalid_config},
        )
        assert r.status_code == 422
        assert r.json()["detail"] == "Invalid input"

    def test_bifurcation_rejected_for_non_positive_timestep(self, client: TestClient) -> None:
        r = client.post(
            "/api/bifurcation",
            json={
                "model_name": MODEL,
                "duration": 20.0,
                "dt": 0.0,
                "sweep_param": "v_rest",
                "sweep_min": -75,
                "sweep_max": -55,
                "sweep_steps": 5,
            },
        )
        assert r.status_code == 422
        assert r.json()["detail"]["limit"] == "timestep"

    def test_normal_sensitivity_within_default_budget_passes(self, client: TestClient) -> None:
        # The default budget admits ordinary analysis requests unchanged.
        r = client.post(
            "/api/sensitivity",
            json={"model_name": MODEL, "duration": 20.0, "current": 10.0},
        )
        assert r.status_code == 200
        assert "sensitivities" in r.json()
