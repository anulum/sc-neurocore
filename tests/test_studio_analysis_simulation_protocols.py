# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio analysis simulation protocols

"""Focused suite: TestSimulationProtocols from former test_studio_analysis.py."""

from __future__ import annotations

from typing import Any

import pytest
from tests.studio_analysis_support import (
    _make_current_trace,
    _make_simulate_fn,
    frequency_response,
    heatmap_2d,
    simulate,
)


class TestSimulationProtocols:
    """Stimulus protocols, frequency response and heatmap sweeps."""

    def test_step_protocol(self) -> None:
        """A step protocol is zero before and 30 after its onset."""
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

    def test_ramp_protocol(self) -> None:
        """A ramp protocol rises from zero to about ten."""
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

    def test_sine_protocol_trace_has_ac_current(self) -> None:
        """A sine protocol swings between about -2 and 2 from zero."""
        trace = _make_current_trace("sine", 2.0, 1000, dt=1.0, frequency_hz=10.0)
        assert trace[0] == pytest.approx(0.0)
        assert max(trace) == pytest.approx(2.0, rel=0.02)
        assert min(trace) == pytest.approx(-2.0, rel=0.02)

    def test_a_biased_sine_oscillates_about_its_bias(self) -> None:
        """A biased sine swings between about 5 and 15 from its bias of 10."""
        trace = _make_current_trace("sine", 5.0, 1000, dt=1.0, frequency_hz=10.0, bias=10.0)
        assert trace[0] == pytest.approx(10.0)
        assert max(trace) == pytest.approx(15.0, rel=0.02)
        assert min(trace) == pytest.approx(5.0, rel=0.02)

    def test_a_bias_is_refused_for_any_protocol_but_the_sine(self) -> None:
        """A bias on a protocol other than the sine is rejected with ValueError."""
        with pytest.raises(ValueError, match="sine protocol only"):
            _make_current_trace("constant", 5.0, 10, bias=1.0)

    def test_frequency_response_drives_a_biased_sine_on_a_real_run(self) -> None:
        """The frequency response drives a biased sine and reports positive rates and gains."""
        # A real LIF equation run: the drive stays positive about its bias.
        result = frequency_response(
            _make_simulate_fn(
                {
                    "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
                    "threshold": "v > -50",
                    "reset": "v = -65",
                    "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
                    "init": {"v": -65.0},
                    "dt": 0.1,
                    "duration": 400.0,
                }
            ),
            {"dt": 0.1, "duration": 400.0},
            freq_min=5.0,
            freq_max=40.0,
            n_freqs=3,
            bias=30.0,
        )
        assert result["bias"] == 30.0 and result["depth"] == 0.5 and result["amplitude"] == 15.0
        assert result["cycles"] == [2, 5, 16]
        assert all(rate is not None and rate > 0 for rate in result["rates"])
        assert all(gain is not None and gain > 0 for gain in result["gain"])
        assert result["contract"]["domain"] == "complete"

    def test_heatmap_2d_returns_failure_metadata_on_success(self) -> None:
        """A complete heatmap sweep reports no failed points and the expected rates."""

        def fake_simulate(**cfg: Any) -> dict[str, Any]:
            """Stand in for a simulation and return a synthetic rate."""
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

    def test_heatmap_2d_fails_closed_with_diagnostics(self) -> None:
        """A heatmap sweep with a failing point raises with its diagnostics."""

        def fake_simulate(**cfg: Any) -> dict[str, Any]:
            """Stand in for a simulation and return a synthetic rate."""
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

    def test_stats_have_isi_histogram(self) -> None:
        """Simulation statistics include an ISI histogram with counts."""
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
