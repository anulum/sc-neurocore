# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: JansenRitUnit

"""Full pipeline test for JansenRitUnit (Jansen & Rit 1995).

Neural mass model for EEG generation. 6 ODEs, 3 populations.
Returns continuous EEG signal (y1-y2), not binary spikes."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.jansen_rit import JansenRitUnit
from sc_neurocore.network.population import Population


class TestJRIsolation:
    def test_construction(self):
        n = JansenRitUnit()
        assert n.y0 == 0.0
        assert n.c == 135.0

    def test_step_returns_float(self):
        n = JansenRitUnit()
        v = n.step()
        assert isinstance(v, float)

    def test_default_drive_oscillates(self):
        """p_ext=220 should produce oscillatory EEG (std > 0)."""
        n = JansenRitUnit()
        vals = [n.step(220.0) for _ in range(10000)]
        assert np.std(vals) > 0.5

    def test_eeg_bounded(self):
        """Output should stay in physiological range."""
        n = JansenRitUnit()
        vals = [n.step(220.0) for _ in range(10000)]
        assert max(vals) < 50.0
        assert min(vals) > -50.0

    def test_zero_drive_stable(self):
        """No external drive → settle to fixed point."""
        n = JansenRitUnit()
        vals = [n.step(0.0) for _ in range(10000)]
        assert np.std(vals[-1000:]) < 1.0

    def test_six_state_variables(self):
        n = JansenRitUnit()
        for _ in range(1000):
            n.step(220.0)
        assert n.y0 != 0.0
        assert n.y1 != 0.0
        assert n.y3 != 0.0

    def test_sigmoid(self):
        n = JansenRitUnit()
        assert n._sigmoid(0.0) < n._sigmoid(10.0)
        assert 0 < n._sigmoid(0.0) < 2 * n.e0

    def test_drive_affects_output(self):
        n_low = JansenRitUnit()
        n_high = JansenRitUnit()
        vals_low = [n_low.step(50.0) for _ in range(5000)]
        vals_high = [n_high.step(300.0) for _ in range(5000)]
        assert np.mean(vals_high[-1000:]) != np.mean(vals_low[-1000:])

    def test_numerical_stability(self):
        for p in [0.0, 100.0, 220.0, 500.0]:
            n = JansenRitUnit()
            for _ in range(10000):
                v = n.step(p)
            for attr in ["y0", "y1", "y2", "y3", "y4", "y5"]:
                assert np.isfinite(getattr(n, attr)), f"{attr} NaN at p={p}"

    def test_reset(self):
        n = JansenRitUnit()
        for _ in range(5000):
            n.step(220.0)
        n.reset()
        assert n.y0 == 0.0
        assert n.y1 == 0.0
        assert n.y5 == 0.0

    def test_deterministic(self):
        n1 = JansenRitUnit()
        n2 = JansenRitUnit()
        for _ in range(500):
            v1 = n1.step(220.0)
            v2 = n2.step(220.0)
            assert v1 == v2


class TestJRNetwork:
    def test_population(self):
        assert Population(JansenRitUnit, n=5, label="jr").n == 5
