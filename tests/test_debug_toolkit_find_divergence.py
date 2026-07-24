# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFindDivergence from former test_debug_toolkit.py

"""Focused suite: TestFindDivergence from former test_debug_toolkit.py."""

from __future__ import annotations

from tests.debug_toolkit_support import *  # noqa: F403


class TestFindDivergence:
    def test_identical_traces(self):
        spikes = np.zeros((10, 5), dtype=np.int8)
        spikes[3, 2] = 1
        t1 = _make_trace(spikes=spikes.copy())
        t2 = _make_trace(spikes=spikes.copy())
        assert find_divergence(t1, t2) is None

    def test_divergent_traces(self):
        s1 = np.zeros((10, 5), dtype=np.int8)
        s2 = np.zeros((10, 5), dtype=np.int8)
        s1[3, 2] = 1
        s2[3, 2] = 0
        v = np.random.randn(10, 5)
        t1 = _make_trace(spikes=s1, voltages=v.copy())
        t2 = _make_trace(spikes=s2, voltages=v.copy())
        dp = find_divergence(t1, t2)
        assert isinstance(dp, DivergencePoint)
        assert dp.timestep == 3
        assert dp.neuron_id == 2
        assert dp.trace_a_spike == 1
        assert dp.trace_b_spike == 0

    def test_different_sizes(self):
        s1 = np.zeros((10, 5), dtype=np.int8)
        s2 = np.zeros((8, 3), dtype=np.int8)
        s1[0, 0] = 1
        t1 = _make_trace(n_neurons=5, n_steps=10, spikes=s1)
        t2 = ExecutionTrace(
            n_neurons=3,
            n_steps=8,
            spikes=s2,
            voltages=np.zeros((8, 3)),
            currents=np.zeros((8, 3)),
        )
        dp = find_divergence(t1, t2)
        assert dp is not None
        assert dp.timestep == 0
