# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExactLIFSolver from former test_solvers_ode.py

"""Focused suite: TestExactLIFSolver from former test_solvers_ode.py."""

from __future__ import annotations

from tests.solvers_ode_support import *  # noqa: F403


class TestExactLIFSolver:
    def test_spike_time_matches_analytical(self):
        solver = ExactLIFSolver(tau=10.0, v_rest=-65.0, v_thresh=-50.0, r_m=1.0)
        # V_inf = -65 + 20 = -45 (above threshold)
        t = solver.next_spike_time(v0=-65.0, current=20.0)
        assert t is not None
        v_at_t = solver.evolve_to_time(-65.0, t, 20.0)
        assert abs(v_at_t - solver.v_thresh) < 1e-8

    def test_subthreshold_no_spike(self):
        solver = ExactLIFSolver(tau=10.0, v_rest=-65.0, v_thresh=-50.0, r_m=1.0)
        t = solver.next_spike_time(v0=-65.0, current=10.0)
        assert t is None  # V_inf = -55, never reaches -50

    def test_already_threshold_spikes_immediately(self):
        solver = ExactLIFSolver(v_thresh=-50.0)

        assert solver.next_spike_time(v0=-50.0, current=20.0) == 0.0

    def test_evolve_to_time_at_zero(self):
        solver = ExactLIFSolver()
        v = solver.evolve_to_time(v0=-60.0, t=0.0, current=0.0)
        assert v == pytest.approx(-60.0)

    def test_subthreshold_evolution_is_bounded_by_equilibrium(self):
        solver = ExactLIFSolver(tau=20.0, v_rest=-65.0, v_thresh=-50.0, r_m=1.0)

        voltage = solver.evolve_to_time(v0=-65.0, t=20.0, current=10.0)

        assert -65.0 < voltage < -55.0
        assert voltage < solver.v_thresh

    def test_firing_rate_suprathreshold(self):
        solver = ExactLIFSolver(tau=10.0, v_rest=-65.0, v_thresh=-50.0, v_reset=-65.0, r_m=1.0)
        rate = solver.firing_rate(current=30.0)
        assert rate > 0

    def test_firing_rate_subthreshold(self):
        solver = ExactLIFSolver(tau=10.0, v_rest=-65.0, v_thresh=-50.0, r_m=1.0)
        rate = solver.firing_rate(current=5.0)
        assert rate == 0.0

    def test_simulate_produces_spikes(self):
        solver = ExactLIFSolver(tau=10.0, v_rest=-65.0, v_thresh=-50.0, v_reset=-65.0, r_m=1.0)
        spikes, _ = solver.simulate(current=30.0, t_end=100.0)
        assert len(spikes) >= 2

    def test_simulate_breaks_when_next_spike_exceeds_window(self):
        solver = ExactLIFSolver(tau=10.0, v_rest=-65.0, v_thresh=-50.0, v_reset=-65.0, r_m=1.0)

        spikes, voltages = solver.simulate(current=20.0, t_end=1.0)

        assert spikes == []
        assert voltages == []

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"tau": 0.0}, "tau"),
            ({"tau": True}, "tau"),
            ({"tau": "bad"}, "tau"),
            ({"v_rest": float("nan")}, "v_rest"),
            ({"v_thresh": float("inf")}, "v_thresh"),
            ({"v_reset": -50.0, "v_thresh": -50.0}, "v_reset"),
            ({"r_m": 0.0}, "r_m"),
        ],
    )
    def test_rejects_invalid_physical_parameters(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            ExactLIFSolver(**kwargs)

    @pytest.mark.parametrize(
        ("method", "args", "match"),
        [
            ("evolve_to_time", {"v0": float("nan"), "t": 1.0, "current": 1.0}, "v0"),
            ("evolve_to_time", {"v0": -65.0, "t": -1.0, "current": 1.0}, "t"),
            ("evolve_to_time", {"v0": -65.0, "t": True, "current": 1.0}, "t"),
            ("next_spike_time", {"v0": -65.0, "current": float("inf")}, "current"),
            ("simulate", {"current": 20.0, "t_end": -1.0}, "t_end"),
            ("simulate", {"current": True, "t_end": 1.0}, "current"),
            ("simulate", {"current": 20.0, "t_end": 1.0, "v0": float("nan")}, "v0"),
        ],
    )
    def test_rejects_invalid_runtime_inputs(self, method, args, match):
        solver = ExactLIFSolver()

        with pytest.raises(ValueError, match=match):
            getattr(solver, method)(**args)
