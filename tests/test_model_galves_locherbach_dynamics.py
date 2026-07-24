# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDynamics from former test_model_galves_locherbach.py

"""Focused suite: TestDynamics from former test_model_galves_locherbach.py."""

from __future__ import annotations

from tests.model_galves_locherbach_support import *  # noqa: F403


class TestDynamics:
    def test_fires_at_test_current(self):
        n = GalvesLocherbachNeuron()
        spikes = _run(n, current=10.0, steps=5000)
        assert len(spikes) >= 50

    def test_rate_increases_with_current(self):
        n_low = GalvesLocherbachNeuron()
        n_high = GalvesLocherbachNeuron()
        s_low = len(_run(n_low, current=0.5, steps=5000))
        s_high = len(_run(n_high, current=10.0, steps=5000))
        assert s_high >= s_low

    def test_two_runs_differ(self):
        n1 = GalvesLocherbachNeuron()
        n2 = GalvesLocherbachNeuron()
        t1 = [n1.step(0.5) for _ in range(1000)]
        t2 = [n2.step(0.5) for _ in range(1000)]
        assert t1 != t2

    def test_logistic_probability_is_bounded_without_overflow_warning(self):
        n = GalvesLocherbachNeuron(v=-1.0e9, steepness=50.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            probability = n._firing_prob()
        assert probability == pytest.approx(0.0)
