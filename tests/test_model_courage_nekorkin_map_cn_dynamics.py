# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCNDynamics from former test_model_courage_nekorkin_map.py

"""Focused suite: TestCNDynamics from former test_model_courage_nekorkin_map.py."""

from __future__ import annotations

from tests.model_courage_nekorkin_map_support import *  # noqa: F403

class TestCNDynamics:
    def test_sustained_bounded_spiking(self):
        """Default regime fires repeatedly and stays bounded (no clip-pegging)."""
        n = CourageNekorkinMapNeuron()
        trace, spikes = n.simulate(20_000, backend="python")
        assert spikes > 1000
        assert np.all(np.isfinite(trace))
        assert trace.max() - trace.min() < 5.0

    def test_burst_structure(self):
        """ISI distribution shows both in-burst (short) and inter-burst (long) gaps."""
        n = CourageNekorkinMapNeuron()
        prev = n.x
        times = []
        for t in range(20_000):
            if n.step(0.0) == 1:
                times.append(t)
            prev = n.x
        _ = prev
        intervals = np.diff(times)
        assert intervals.min() <= 3  # in-burst spikes
        assert intervals.max() >= 8  # inter-burst quiescence

    def test_chaos_sensitivity(self):
        """Tiny initial offset amplifies — sensitive dependence on initial conditions."""
        a = CourageNekorkinMapNeuron(x=0.0)
        b = CourageNekorkinMapNeuron(x=1e-9)
        tr_a, _ = a.simulate(2000, backend="python")
        tr_b, _ = b.simulate(2000, backend="python")
        assert abs(tr_a[-1] - tr_b[-1]) > 1e-3

    def test_quiescent_below_threshold_regime(self):
        """J < Jmin gives the excitable (non-spiking-bursting) regime — far fewer spikes."""
        jmin, _ = _breakpoints()
        excitable = CourageNekorkinMapNeuron(j=jmin - 0.05)
        _, spikes = excitable.simulate(20_000, backend="python")
        _, spikes_default = CourageNekorkinMapNeuron().simulate(20_000, backend="python")
        assert spikes < spikes_default

    @pytest.mark.parametrize("current", [-0.02, 0.0, 0.05, 0.1])
    def test_fi_sweep_finite(self, current: float):
        n = CourageNekorkinMapNeuron()
        trace, _ = n.simulate(5000, current, backend="python")
        assert np.all(np.isfinite(trace))

    def test_upward_crossing_only(self):
        n = CourageNekorkinMapNeuron()
        prev_x = n.x
        for _ in range(5000):
            spike = n.step(0.0)
            if spike == 1:
                assert prev_x < n.x_threshold
            prev_x = n.x
