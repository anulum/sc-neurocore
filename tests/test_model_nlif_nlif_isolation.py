# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNLIFIsolation from former test_model_nlif.py

"""Focused suite: TestNLIFIsolation from former test_model_nlif.py."""

from __future__ import annotations

from tests.model_nlif_support import *  # noqa: F403


class TestNLIFIsolation:
    def test_defaults(self):
        n = NonlinearLIFNeuron()
        assert n.v == -65.0 and n.w == 0.0
        assert n.v_rest == -65.0 and n.v_crit == -40.0
        assert n.v_threshold == -20.0 and n.a == 0.04
        assert n.b == 0.5 and n.tau_w == 100.0 and n.c_m == 1.0

    def test_step_returns_binary(self):
        assert NonlinearLIFNeuron().step(0.0) in (0, 1)

    def test_both_states_evolve(self):
        n = NonlinearLIFNeuron()
        v0, w0 = n.v, n.w
        for _ in range(500):
            n.step(20.0)
        assert n.v != v0

    def test_state_finite_long_run(self):
        n = NonlinearLIFNeuron()
        for _ in range(100_000):
            n.step(20.0)
        assert np.isfinite(n.v) and np.isfinite(n.w)

    def test_reset_restores_defaults(self):
        n = NonlinearLIFNeuron(v_rest=-62.0, v_reset=-58.0, v_crit=-40.0, v_threshold=-20.0)
        for _ in range(5000):
            n.step(20.0)
        n.reset()
        assert n.v == n.v_rest and n.w == 0.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"v": np.nan},
            {"w": np.inf},
            {"v_rest": np.nan},
            {"v_crit": np.inf},
            {"v_threshold": np.nan},
            {"v_reset": np.inf},
            {"v_crit": -70.0},
            {"v_threshold": -45.0},
            {"v_reset": -10.0},
            {"a": -0.01},
            {"a": np.nan},
            {"b": -0.1},
            {"tau_w": 0.0},
            {"c_m": 0.0},
            {"dt": 0.0},
            {"dt": 101.0},
        ],
    )
    def test_rejects_non_physical_configuration(self, kwargs):
        with pytest.raises(ValueError):
            NonlinearLIFNeuron(**kwargs)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float):
        n = NonlinearLIFNeuron(v=-60.0, w=0.5)
        before = (n.v, n.w)
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert (n.v, n.w) == before

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = NonlinearLIFNeuron()
            trace = [(n.step(20.0), n.v, n.w) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]
