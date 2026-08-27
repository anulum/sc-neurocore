# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLBIsolation from former test_model_larter_breakspear.py

"""Focused suite: TestLBIsolation from former test_model_larter_breakspear.py."""

from __future__ import annotations

from tests.model_larter_breakspear_support import *  # noqa: F403


class TestLBIsolation:
    def test_defaults(self):
        n = LarterBreakspearNeuron()
        assert n.v == 0.1 and n.w == 0.1 and n.z == 0.1
        assert n.g_ca == 1.1 and n.g_na == 6.7
        assert n.a_ee == 0.4 and n.r_nmda == 0.25
        assert n.coupling_balance == 0.1
        assert n.dt == 0.01
        assert n.integrator == "rk4"

    def test_step_returns_float(self):
        n = LarterBreakspearNeuron()
        result = n.step(0.0)
        assert isinstance(result, (float, np.floating))

    def test_state_finite_long_run(self):
        n = LarterBreakspearNeuron()
        for _ in range(50_000):
            n.step(0.0)
        assert np.isfinite(n.v) and np.isfinite(n.w) and np.isfinite(n.z)

    def test_reset_restores_defaults(self):
        n = LarterBreakspearNeuron()
        for _ in range(5000):
            n.step(0.0)
        n.reset()
        assert n.v == 0.1 and n.w == 0.1 and n.z == 0.1

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = LarterBreakspearNeuron()
            trace = [n.step(0.0) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]

    def test_rejects_nonfinite_coupling(self):
        n = LarterBreakspearNeuron()
        with pytest.raises(ValueError, match="coupling"):
            n.step(float("inf"))
