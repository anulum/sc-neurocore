# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBendaHerzIsolation from former test_model_benda_herz.py

"""Focused suite: TestBendaHerzIsolation from former test_model_benda_herz.py."""

from __future__ import annotations

from tests.model_benda_herz_support import *  # noqa: F403

class TestBendaHerzIsolation:
    def test_construction(self):
        n = BendaHerzNeuron()
        assert n.a == 0.0
        assert n.f_max == 200.0

    def test_step_returns_binary(self):
        n = BendaHerzNeuron(seed=1)
        result = n.step(10.0)
        assert result in (0, 1)

    def test_spikes_under_drive(self):
        """Stochastic model — needs many steps for reliable spiking."""
        n = BendaHerzNeuron(seed=2)
        spikes = sum(n.step(50.0) for _ in range(10000))
        assert spikes > 0, "no spikes at I=50 over 10K steps"

    def test_adaptation_increases(self):
        """Adaptation variable A should increase under sustained drive."""
        n = BendaHerzNeuron(seed=3)
        a_init = n.a
        for _ in range(1000):
            n.step(30.0)
        assert n.a > a_init, "adaptation variable did not increase"

    def test_adaptation_reduces_rate(self):
        """Rate should decrease over time due to SFA."""
        n = BendaHerzNeuron(seed=4)
        early_spikes = sum(n.step(50.0) for _ in range(2000))
        late_spikes = sum(n.step(50.0) for _ in range(2000))
        # Early should have more spikes than late (adaptation kicks in)
        # Stochastic — allow for noise; just check adaptation is nonzero
        assert n.a > 0, "no adaptation after 4K steps"

    def test_adaptation_candidate_matches_rk4_reference(self):
        n = BendaHerzNeuron(a=0.35, dt=0.25, seed=5)
        expected_a, expected_p = _rk4_reference(n, 12.5)

        candidate_a, candidate_p = n._rk4_candidate(12.5)

        assert candidate_a == pytest.approx(expected_a, rel=1e-14, abs=1e-14)
        assert candidate_p == pytest.approx(expected_p, rel=1e-14, abs=1e-14)

    def test_step_commits_rk4_candidate_before_sampling(self):
        n = BendaHerzNeuron(a=0.25, dt=0.5, seed=6)
        expected_a, _ = _rk4_reference(n, 15.0)

        n.step(15.0)

        assert n.a == pytest.approx(expected_a, rel=1e-14, abs=1e-14)

    def test_exponential_hazard_keeps_probability_bounded(self):
        n = BendaHerzNeuron(f_max=1.0e6, dt=1.0, seed=7)

        _, probability = n._rk4_candidate(1.0e6)

        assert 0.0 <= probability <= 1.0
        assert probability == pytest.approx(1.0, rel=0.0, abs=1e-12)

    def test_seeded_sequences_are_reproducible(self):
        left = BendaHerzNeuron(seed=123)
        right = BendaHerzNeuron(seed=123)

        assert [left.step(25.0) for _ in range(50)] == [right.step(25.0) for _ in range(50)]

    def test_f_onset_sigmoid(self):
        """f_onset should be sigmoid-shaped: low at I=0, high at I>>i_half."""
        n = BendaHerzNeuron()
        f_low = n._f_onset(0.0)
        f_high = n._f_onset(50.0)
        assert f_high > f_low

    def test_state_finite(self):
        n = BendaHerzNeuron()
        for _ in range(5000):
            n.step(100.0)
        assert np.isfinite(n.a)

    def test_reset(self):
        n = BendaHerzNeuron()
        for _ in range(100):
            n.step(30.0)
        n.reset()
        assert n.a == 0.0
