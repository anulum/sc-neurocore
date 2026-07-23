# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestButeraIsolation from former test_model_butera_respiratory.py

"""Focused suite: TestButeraIsolation from former test_model_butera_respiratory.py."""

from __future__ import annotations

from tests.model_butera_respiratory_support import *  # noqa: F403

class TestButeraIsolation:
    def test_construction(self):
        n = ButeraRespiratoryNeuron()
        assert n.v == -50.0
        assert n.h_nap == 0.5

    def test_step_returns_binary(self):
        n = ButeraRespiratoryNeuron()
        assert n.step(0.0) in (0, 1)

    def test_subthreshold(self):
        n = ButeraRespiratoryNeuron()
        spikes = sum(n.step(10.0) for _ in range(10_000))
        assert spikes == 0

    def test_spikes_at_high_current(self):
        n = ButeraRespiratoryNeuron()
        spikes = sum(n.step(100.0) for _ in range(100_000))
        assert spikes > 100, f"too few spikes at I=100: {spikes}"

    def test_persistent_na_inactivation(self):
        """h_nap should change from initial value under sustained drive."""
        n = ButeraRespiratoryNeuron()
        h_init = n.h_nap
        for _ in range(100_000):
            n.step(100.0)
        assert n.h_nap != h_init

    def test_numerical_stability(self):
        for I in [0, 10, 50, 100]:
            n = ButeraRespiratoryNeuron()
            for _ in range(50_000):
                n.step(float(I))
            assert np.isfinite(n.v), f"v NaN at I={I}"
            assert np.isfinite(n.n), f"n NaN at I={I}"
            assert np.isfinite(n.h_nap), f"h_nap NaN at I={I}"

    def test_gating_bounded(self):
        n = ButeraRespiratoryNeuron()
        for _ in range(50_000):
            n.step(100.0)
        assert 0 <= n.n <= 1
        assert 0 <= n.h_nap <= 1

    def test_reset(self):
        n = ButeraRespiratoryNeuron()
        for _ in range(1000):
            n.step(100.0)
        n.reset()
        assert n.v == -50.0
        assert n.n == 0.01
        assert n.h_nap == 0.5
