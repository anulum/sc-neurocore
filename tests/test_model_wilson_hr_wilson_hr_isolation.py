# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWilsonHRIsolation from former test_model_wilson_hr.py

"""Focused suite: TestWilsonHRIsolation from former test_model_wilson_hr.py."""

from __future__ import annotations

from tests.model_wilson_hr_support import *  # noqa: F403

class TestWilsonHRIsolation:
    def test_defaults(self):
        n = WilsonHRNeuron()
        assert n.v == -0.7
        assert n.r == 0.1
        assert n.tau_r == 1.9
        assert n.v_peak == 0.4
        assert n.dt == 0.05

    def test_step_returns_binary(self):
        assert WilsonHRNeuron().step(0.0) in (0, 1)

    def test_two_variables_evolve(self):
        n = WilsonHRNeuron()
        v0, r0 = n.v, n.r
        for _ in range(100):
            n.step(0.3)
        assert n.v != v0 and n.r != r0

    def test_state_finite(self):
        n = WilsonHRNeuron()
        for _ in range(50_000):
            n.step(0.3)
        assert np.isfinite(n.v) and np.isfinite(n.r)

    def test_reset(self):
        n = WilsonHRNeuron()
        for _ in range(100):
            n.step(0.3)
        n.reset()
        assert n.v == -0.7 and n.r == 0.1

    def test_spike_resets_v(self):
        n = WilsonHRNeuron()
        for _ in range(50_000):
            if n.step(0.3) == 1:
                assert n.v == -0.7
                break
