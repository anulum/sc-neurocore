# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPerfectIntegratorEdgeCases from former test_model_perfect_integrator.py

"""Focused suite: TestPerfectIntegratorEdgeCases from former test_model_perfect_integrator.py."""

from __future__ import annotations

from tests.model_perfect_integrator_support import *  # noqa: F403

class TestPerfectIntegratorEdgeCases:
    """Numerical edge cases and parameter boundaries."""

    def test_negative_current_no_spike(self):
        """Negative input drives V below reset — never reaches threshold."""
        n = PerfectIntegratorNeuron()
        spikes = sum(n.step(-5.0) for _ in range(1000))
        assert spikes == 0
        assert n.v < 0.0

    def test_large_negative_current_finite(self):
        """Even extreme negative current keeps V finite."""
        n = PerfectIntegratorNeuron()
        for _ in range(100000):
            n.step(-1e6)
        assert np.isfinite(n.v)

    def test_very_small_dt(self):
        """Fine time resolution: more steps to spike, same total time."""
        n = PerfectIntegratorNeuron(dt=0.001)
        I = 5.0
        # Analytical: steps = θ / (I/C * dt) = 1.0 / 0.005 = 200
        times = _collect_spike_times(n, current=I, steps=500)
        assert len(times) >= 1
        assert abs(times[0] - 200) <= 1

    def test_very_large_dt(self):
        """Coarse dt: spike on first step if dV >= θ."""
        n = PerfectIntegratorNeuron(dt=1.0)
        # dV = 5.0 * 1.0 = 5.0 >= 1.0
        assert n.step(5.0) == 1

    def test_threshold_must_exceed_reset(self):
        """Zero threshold excursion is a degenerate no-distance ISI."""
        with pytest.raises(ValueError, match="v_threshold"):
            PerfectIntegratorNeuron(v_threshold=0.0, v_reset=0.0)

    def test_floating_point_accumulation(self):
        """Document fp rounding: 10 additions of 0.1 ≠ 1.0 exactly.

        With I=1.0, dt=0.1, C=1.0: dV=0.1/step. After 10 steps,
        V = 0.99999... due to IEEE 754 rounding of 0.1.
        Spike is delayed to step 11.
        """
        n = PerfectIntegratorNeuron()
        times = _collect_spike_times(n, current=1.0, steps=15)
        assert len(times) >= 1
        # Spike at step 10 or 11 depending on fp accumulation
        assert times[0] in (9, 10), f"First spike at step {times[0]}"

    def test_alternating_current(self):
        """Alternating +/- current: voltage oscillates around 0, no spikes."""
        n = PerfectIntegratorNeuron()
        spikes = 0
        for t in range(10000):
            sign = 1.0 if t % 2 == 0 else -1.0
            spikes += n.step(sign * 3.0)
        assert spikes == 0
        assert abs(n.v) < 1e-10

    def test_reset_method(self):
        n = PerfectIntegratorNeuron()
        for _ in range(50):
            n.step(3.0)
        n.reset()
        assert n.v == n.v_reset

    def test_deterministic_reproducibility(self):
        """Exact bit-for-bit reproducibility across runs."""
        runs = []
        for _ in range(3):
            n = PerfectIntegratorNeuron()
            trace = [(n.step(3.5), n.v) for _ in range(200)]
            runs.append(trace)
        assert runs[0] == runs[1] == runs[2]
