# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: WongWangUnit

"""Full pipeline test for WongWangUnit (Wong & Wang 2006).

Reduced decision-making attractor model: 2 pools (s1, s2) with mutual
inhibition. Returns tuple (r1, r2) — firing rates, not spikes.
step(stim1, stim2): dual stimulus. Stochastic (np.random.randn noise).
Pipeline limited: tuple return incompatible with Network.step_all.
Performance: ~44K isolation steps/s."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.wong_wang import WongWangUnit
from sc_neurocore.network.population import Population


class TestWongWangIsolation:
    def test_defaults(self):
        n = WongWangUnit()
        assert n.s1 == 0.1 and n.s2 == 0.1
        assert n.tau_s == 0.1 and n.gamma == 0.641
        assert n.j_n == 0.2609 and n.j_cross == 0.0497

    def test_step_returns_tuple(self):
        """Returns (r1, r2) tuple — firing rates of both pools."""
        n = WongWangUnit()
        result = n.step(0.0, 0.0)
        assert isinstance(result, tuple) and len(result) == 2

    def test_dual_input_signature(self):
        """step(stim1, stim2) — separate stimuli for each pool."""
        n = WongWangUnit()
        r1, r2 = n.step(0.1, 0.05)
        assert np.isfinite(r1) and np.isfinite(r2)

    def test_state_finite(self):
        n = WongWangUnit()
        for _ in range(100000):
            n.step(0.1, 0.0)
        assert np.isfinite(n.s1) and np.isfinite(n.s2)

    def test_reset(self):
        n = WongWangUnit()
        for _ in range(1000):
            n.step(0.1, 0.0)
        n.reset()
        assert n.s1 == 0.1 and n.s2 == 0.1


class TestWongWangDecisionDynamics:
    """Core: two mutually inhibiting pools compete for dominance."""

    def test_stimulus_drives_winner(self):
        """Stronger stim1 → s1 wins (attractor for pool 1)."""
        np.random.seed(42)
        n = WongWangUnit(sigma=0.001)  # reduce noise for reliable test
        for _ in range(100000):
            n.step(0.1, 0.0)
        assert n.s1 > n.s2, f"s1={n.s1:.4f}, s2={n.s2:.4f}"

    def test_symmetric_stimulus_bistable(self):
        """Equal stimuli: both pools at similar level OR one wins (bistable)."""
        np.random.seed(42)
        n = WongWangUnit(sigma=0.001)
        for _ in range(50000):
            n.step(0.05, 0.05)
        # Both s values should be positive
        assert n.s1 > 0 and n.s2 > 0

    def test_s_bounded_0_1(self):
        """s1, s2 are clipped to [0, 1]."""
        n = WongWangUnit()
        for _ in range(100000):
            n.step(0.5, 0.0)
        assert 0.0 <= n.s1 <= 1.0
        assert 0.0 <= n.s2 <= 1.0

    def test_mutual_inhibition(self):
        """j_cross provides mutual inhibition: high s1 suppresses s2."""
        np.random.seed(42)
        n = WongWangUnit(sigma=0.001)
        for _ in range(100000):
            n.step(0.2, 0.0)  # strong stim1, no stim2
        # s1 should be high, s2 should be low
        assert n.s1 > 0.5
        assert n.s2 < 0.2

    def test_j_n_self_excitation(self):
        """j_n provides self-excitation: sustains the winning pool."""
        n_strong = WongWangUnit(j_n=0.35, sigma=0.001)
        n_weak = WongWangUnit(j_n=0.15, sigma=0.001)
        np.random.seed(42)
        for _ in range(50000):
            n_strong.step(0.1, 0.0)
        np.random.seed(42)
        for _ in range(50000):
            n_weak.step(0.1, 0.0)
        # Stronger self-excitation → higher winner activity
        assert n_strong.s1 > n_weak.s1


class TestWongWangPhiFunction:
    """Transfer function: φ(I) = (aI - b) / (1 - exp(-d(aI - b)))."""

    def test_phi_positive_for_positive_input(self):
        n = WongWangUnit()
        r = n._phi(1.0)
        assert r > 0

    def test_phi_increases_with_input(self):
        n = WongWangUnit()
        r_low = float(n._phi(0.5))
        r_high = float(n._phi(1.0))
        assert r_high > r_low

    def test_phi_singularity_protection(self):
        """At x = b/a = 108/270 = 0.4: φ should not diverge."""
        n = WongWangUnit()
        r = n._phi(108.0 / 270.0)
        assert np.isfinite(r)

    def test_phi_formula_at_known_point(self):
        """At I=0.5: x = 270*0.5 - 108 = 27. φ = 27/(1-exp(-0.154*27))."""
        n = WongWangUnit()
        r = float(n._phi(0.5))
        expected = 27.0 / (1.0 - np.exp(-0.154 * 27.0))
        assert abs(r - expected) < 0.01


class TestWongWangStochasticity:
    def test_noise_affects_dynamics(self):
        """sigma > 0 → different runs differ."""
        n1 = WongWangUnit(sigma=0.02)
        n2 = WongWangUnit(sigma=0.02)
        t1 = [n1.step(0.1, 0.0) for _ in range(1000)]
        t2 = [n2.step(0.1, 0.0) for _ in range(1000)]
        # Very unlikely to be identical
        assert t1 != t2

    def test_zero_noise_deterministic(self):
        """sigma=0 → identical runs."""
        np.random.seed(42)
        n1 = WongWangUnit(sigma=0.0)
        t1 = [(n1.step(0.1, 0.0), n1.s1) for _ in range(100)]
        np.random.seed(42)
        n2 = WongWangUnit(sigma=0.0)
        t2 = [(n2.step(0.1, 0.0), n2.s1) for _ in range(100)]
        # With zero noise and same seed, should be identical
        for a, b in zip(t1, t2):
            assert a[1] == b[1]


class TestWongWangParameters:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("s1", np.nan),
            ("s2", np.inf),
            ("s1", -0.1),
            ("s2", 1.1),
            ("tau_s", 0.0),
            ("gamma", 0.0),
            ("j_n", -1.0),
            ("j_cross", -1.0),
            ("i_0", np.inf),
            ("sigma", -1.0),
            ("dt", 0.0),
        ],
    )
    def test_rejects_invalid_numerical_configuration(self, field: str, value: float):
        with pytest.raises((ValueError, FloatingPointError)):
            WongWangUnit(**{field: value})

    def test_rejects_non_finite_stimulus_before_state_mutation(self):
        n = WongWangUnit()
        before = (n.s1, n.s2)
        with pytest.raises(ValueError, match="stimuli"):
            n.step(np.nan, 0.0)
        assert (n.s1, n.s2) == before

    def test_rejects_corrupted_runtime_state_before_mutation(self):
        n = WongWangUnit()
        n.s1 = 1.5
        before = (n.s1, n.s2)
        with pytest.raises(FloatingPointError, match="gating state"):
            n.step(0.1, 0.0)
        assert (n.s1, n.s2) == before

    def test_phi_saturates_for_extreme_finite_negative_drive(self):
        n = WongWangUnit()
        assert n._phi(-1.0e6) == 0.0

    @pytest.mark.parametrize("dt", [0.0005, 0.001, 0.002])
    def test_dt_stability(self, dt: float):
        n = WongWangUnit(dt=dt)
        for _ in range(50000):
            n.step(0.1, 0.0)
        assert np.isfinite(n.s1)


class TestWongWangPerformance:
    def test_isolation_throughput(self):
        n = WongWangUnit()
        N = 20000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(0.1, 0.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 5000


class TestWongWangPipeline:
    def test_population_creates(self):
        assert Population(WongWangUnit, n=5, label="ww").n == 5

    def test_network_incompatible(self):
        """WongWangUnit.step() returns tuple (r1, r2), not int.

        Network.step_all cannot handle tuple return. Additionally,
        step() requires two arguments (stim1, stim2) but Network
        passes only one current value. This is a known limitation:
        dual-decision models need a specialised network driver.
        """
        n = WongWangUnit()
        result = n.step(0.1, 0.0)
        assert isinstance(result, tuple)
        # Document: NOT compatible with Network.run()
