# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMcKeanParameters from former test_model_mckean.py

"""Focused suite: TestMcKeanParameters from former test_model_mckean.py."""

from __future__ import annotations

from tests.model_mckean_support import *  # noqa: F403


class TestMcKeanParameters:
    @pytest.mark.parametrize("epsilon", [0.005, 0.01, 0.05])
    def test_epsilon_timescale(self, epsilon: float):
        n = McKeanNeuron(epsilon=epsilon)
        for _ in range(20_000):
            n.step(0.5)
        assert np.isfinite(n.v) and np.isfinite(n.w)

    @pytest.mark.parametrize("a", [0.1, 0.25, 0.4])
    def test_a_breakpoint_sweep(self, a: float):
        n = McKeanNeuron(a=a)
        for _ in range(20_000):
            n.step(0.5)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("gamma", [0.3, 0.5, 0.8])
    def test_gamma_sweep(self, gamma: float):
        n = McKeanNeuron(gamma=gamma)
        for _ in range(20_000):
            n.step(0.5)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        n = McKeanNeuron(dt=dt)
        for _ in range(20_000):
            n.step(0.5)
        assert np.isfinite(n.v) and np.isfinite(n.w)
