# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBatchAndDispatch from former test_model_adaptive_threshold_if.py

"""Focused suite: TestBatchAndDispatch from former test_model_adaptive_threshold_if.py."""

from __future__ import annotations

from tests.model_adaptive_threshold_if_support import *  # noqa: F403

class TestBatchAndDispatch:
    """The maintained batch lane matches the scalar golden loop."""

    def test_batch_matches_scalar_step_loop(self) -> None:
        drive = 12.0 + 6.0 * np.sin(np.arange(256) * 0.037)
        scalar = AdaptiveThresholdIFNeuron()
        expected_v = []
        expected_theta = []
        expected_spikes = 0
        for value in drive:
            expected_spikes += scalar.step(float(value))
            expected_v.append(scalar.v)
            expected_theta.append(scalar.theta)
        batch = AdaptiveThresholdIFNeuron().simulate(drive, backend="python")
        np.testing.assert_allclose(batch["v"], expected_v, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(batch["theta"], expected_theta, rtol=0.0, atol=0.0)
        assert batch["spike_count"] == expected_spikes

    def test_empty_batch_returns_initial_state(self) -> None:
        result = AdaptiveThresholdIFNeuron(v=-60.0, theta=-48.0).simulate([], backend="python")
        assert result["v"].size == 0
        assert result["v_final"] == -60.0
        assert result["theta_final"] == -48.0
        assert result["spike_count"] == 0

    def test_simulate_writes_back_final_state(self) -> None:
        n = AdaptiveThresholdIFNeuron()
        result = n.simulate(np.full(200, 20.0), backend="python")
        assert n.v == result["v_final"]
        assert n.theta == result["theta_final"]

    def test_long_varied_run_is_finite_and_deterministic(self) -> None:
        drive = 14.0 + 5.0 * np.sin(np.arange(20_000, dtype=np.float64) * 0.013)
        first = AdaptiveThresholdIFNeuron()
        second = AdaptiveThresholdIFNeuron()
        trace_first = first.simulate(drive, backend="python")
        trace_second = second.simulate(drive, backend="python")
        assert np.isfinite(trace_first["v"]).all()
        assert np.isfinite(trace_first["theta"]).all()
        np.testing.assert_array_equal(trace_first["v"], trace_second["v"])
        np.testing.assert_array_equal(trace_first["theta"], trace_second["theta"])
