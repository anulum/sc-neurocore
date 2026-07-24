# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestReservoirDynamics from former test_reservoir.py

"""Focused suite: TestReservoirDynamics from former test_reservoir.py."""

from __future__ import annotations

from tests.reservoir_support import *  # noqa: F403


class TestReservoirDynamics:
    def test_step_output_shape(self):
        res = AutoCriticalReservoir(n_inputs=3, n_neurons=50, seed=0)
        out = res.step(np.array([1.0, 0.0, -1.0]))
        assert out.shape == (50,)
        assert set(np.unique(out)).issubset({0.0, 1.0})

    def test_run_output_shape(self):
        res = AutoCriticalReservoir(n_inputs=2, n_neurons=50, seed=0)
        inputs = np.random.randn(20, 2)
        states = res.run(inputs)
        assert states.shape == (20, 50)

    def test_reset_clears_state(self):
        res = AutoCriticalReservoir(n_inputs=1, n_neurons=20, seed=0)
        res.step(np.array([5.0]))
        res.reset()
        assert np.all(res._v == 0)
        assert np.all(res._spikes == 0)

    def test_run_produces_spikes(self):
        res = AutoCriticalReservoir(n_inputs=1, n_neurons=100, seed=0)
        inputs = np.ones((50, 1)) * 2.0
        states = res.run(inputs)
        assert states.sum() > 0

    def test_different_inputs_different_states(self):
        res = AutoCriticalReservoir(n_inputs=1, n_neurons=50, seed=0)
        s1 = res.run(np.ones((10, 1)) * 2.0)
        s2 = res.run(np.ones((10, 1)) * -2.0)
        assert not np.array_equal(s1, s2)
