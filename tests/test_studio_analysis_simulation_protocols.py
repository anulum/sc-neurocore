# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio analysis simulation protocols

"""Focused suite: TestSimulationProtocols from former test_studio_analysis.py."""

from __future__ import annotations

from tests.studio_analysis_support import *  # noqa: F403

class TestSimulationProtocols:
    def test_step_protocol(self):
        r = simulate(
            equations=["dv/dt = -(v - E_L) / tau_m + I / C"],
            threshold="v > -50",
            reset="v = -65",
            params={"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
            init={"v": -65.0},
            dt=0.1,
            duration=100.0,
            current=30.0,
            protocol="step",
        )
        assert "current_trace" in r
        assert r["current_trace"][0] == 0.0
        mid = len(r["current_trace"]) // 2
        assert r["current_trace"][mid] == 30.0

    def test_ramp_protocol(self):
        r = simulate(
            equations=["dv/dt = I"],
            init={"v": 0.0},
            dt=0.1,
            duration=10.0,
            current=10.0,
            protocol="ramp",
        )
        assert r["current_trace"][0] == 0.0
        assert r["current_trace"][-1] == pytest.approx(10.0, rel=0.1)

    def test_sine_protocol_trace_has_ac_current(self):
        trace = _make_current_trace("sine", 2.0, 1000, dt=1.0, frequency_hz=10.0)
        assert trace[0] == pytest.approx(0.0)
        assert max(trace) == pytest.approx(2.0, rel=0.02)
        assert min(trace) == pytest.approx(-2.0, rel=0.02)

    def test_frequency_response_uses_sine_protocol(self):
        calls: list[dict[str, float | str]] = []

        def fake_simulate(**cfg):
            calls.append(cfg)
            return {"stats": {"rate_hz": float(cfg["frequency_hz"])}}

        result = frequency_response(
            fake_simulate,
            {"dt": 0.1, "duration": 20.0},
            freq_min=5.0,
            freq_max=20.0,
            n_freqs=3,
            amplitude=4.0,
        )
        assert result["rates"] == pytest.approx([5.0, 10.0, 20.0])
        assert all(call["protocol"] == "sine" for call in calls)
        assert all(call["current"] == 4.0 for call in calls)

    def test_heatmap_2d_returns_failure_metadata_on_success(self):
        def fake_simulate(**cfg):
            params = cfg["params"]
            return {"stats": {"rate_hz": float(params["ix"] + params["iy"])}}

        result = heatmap_2d(
            fake_simulate,
            base_config={"params": {"baseline": 1.0}},
            param_x="ix",
            x_min=1.0,
            x_max=2.0,
            x_steps=2,
            param_y="iy",
            y_min=10.0,
            y_max=20.0,
            y_steps=2,
        )
        assert result["failed_points"] == 0
        assert result["total_points"] == 4
        assert result["failure_rate"] == 0.0
        assert result["rates"] == [[11.0, 12.0], [21.0, 22.0]]

    def test_heatmap_2d_fails_closed_with_diagnostics(self):
        def fake_simulate(**cfg):
            params = cfg["params"]
            if params["ix"] == 2.0 and params["iy"] == 20.0:
                raise RuntimeError("synthetic failure")
            return {"stats": {"rate_hz": 1.0}}

        with pytest.raises(ValueError) as exc_info:
            heatmap_2d(
                fake_simulate,
                base_config={"params": {}},
                param_x="ix",
                x_min=1.0,
                x_max=2.0,
                x_steps=2,
                param_y="iy",
                y_min=10.0,
                y_max=20.0,
                y_steps=2,
            )

        err = exc_info.value
        assert "heatmap sweep failed for 1/4 points" in str(err)
        diagnostics = err.args[1]
        assert diagnostics["failed_points"] == 1
        assert diagnostics["total_points"] == 4
        assert diagnostics["failure_rate"] == pytest.approx(0.25)
        assert diagnostics["failures"] == [
            {
                "grid_index": [1, 1],
                "param_x_value": 2.0,
                "param_y_value": 20.0,
                "error_type": "RuntimeError",
                "error_message": "synthetic failure",
            }
        ]

    def test_stats_have_isi_histogram(self):
        r = simulate(
            equations=["dv/dt = -(v - E_L) / tau_m + I / C"],
            threshold="v > -50",
            reset="v = -65",
            params={"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
            init={"v": -65.0},
            dt=0.1,
            duration=200.0,
            current=30.0,
        )
        if r["spike_count"] >= 3:
            assert r["stats"]["isi_histogram"] is not None
            assert "counts" in r["stats"]["isi_histogram"]

