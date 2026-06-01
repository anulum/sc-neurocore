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

import sys
import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.wong_wang import WongWangUnit
from sc_neurocore.network.population import Population


def _rk4_expected_state(n: WongWangUnit, stim1: float, stim2: float) -> tuple[float, float]:
    def rhs(s1: float, s2: float) -> tuple[float, float]:
        r1 = n._phi(n.j_n * s1 - n.j_cross * s2 + n.i_0 + stim1)
        r2 = n._phi(n.j_n * s2 - n.j_cross * s1 + n.i_0 + stim2)
        return (
            -s1 / n.tau_s + (1.0 - s1) * n.gamma * r1,
            -s2 / n.tau_s + (1.0 - s2) * n.gamma * r2,
        )

    s1, s2 = n.s1, n.s2
    k1_1, k1_2 = rhs(s1, s2)
    k2_1, k2_2 = rhs(s1 + 0.5 * n.dt * k1_1, s2 + 0.5 * n.dt * k1_2)
    k3_1, k3_2 = rhs(s1 + 0.5 * n.dt * k2_1, s2 + 0.5 * n.dt * k2_2)
    k4_1, k4_2 = rhs(s1 + n.dt * k3_1, s2 + n.dt * k3_2)
    return (
        min(1.0, max(0.0, s1 + n.dt * (k1_1 + 2.0 * k2_1 + 2.0 * k3_1 + k4_1) / 6.0)),
        min(1.0, max(0.0, s2 + n.dt * (k1_2 + 2.0 * k2_2 + 2.0 * k3_2 + k4_2) / 6.0)),
    )


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
        """Higher stim1 -> s1 wins (attractor for pool 1)."""
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
            n.step(0.2, 0.0)  # high stim1, no stim2
        # s1 should be high, s2 should be low
        assert n.s1 > 0.5
        assert n.s2 < 0.2

    def test_j_n_self_excitation(self):
        """j_n provides self-excitation: sustains the winning pool."""
        n_high = WongWangUnit(j_n=0.35, sigma=0.001)
        n_weak = WongWangUnit(j_n=0.15, sigma=0.001)
        np.random.seed(42)
        for _ in range(50000):
            n_high.step(0.1, 0.0)
        np.random.seed(42)
        for _ in range(50000):
            n_weak.step(0.1, 0.0)
        # Higher self-excitation -> higher winner activity
        assert n_high.s1 > n_weak.s1

    def test_rk4_integrates_full_coupled_ode_not_forward_euler(self):
        n = WongWangUnit(s1=0.24, s2=0.11, sigma=0.0, dt=0.02)
        expected_s1, expected_s2 = _rk4_expected_state(n, 0.17, 0.03)
        old_s1, old_s2 = n.s1, n.s2
        r1 = n._phi(n.j_n * old_s1 - n.j_cross * old_s2 + n.i_0 + 0.17)
        r2 = n._phi(n.j_n * old_s2 - n.j_cross * old_s1 + n.i_0 + 0.03)
        euler_s1 = min(1.0, max(0.0, old_s1 + (-old_s1 / n.tau_s + (1.0 - old_s1) * n.gamma * r1) * n.dt))
        euler_s2 = min(1.0, max(0.0, old_s2 + (-old_s2 / n.tau_s + (1.0 - old_s2) * n.gamma * r2) * n.dt))

        n.step(0.17, 0.03)

        assert n.s1 == pytest.approx(expected_s1, abs=1e-15)
        assert n.s2 == pytest.approx(expected_s2, abs=1e-15)
        assert abs(n.s1 - euler_s1) > 1e-5
        assert abs(n.s2 - euler_s2) > 1e-5


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

    def test_rejects_corrupted_runtime_parameters_before_mutation(self):
        n = WongWangUnit()
        n.dt = 0.0
        before = (n.s1, n.s2)
        with pytest.raises(ValueError, match="dt"):
            n.step(0.1, 0.0)
        assert (n.s1, n.s2) == before

    def test_phi_saturates_for_extreme_finite_negative_drive(self):
        n = WongWangUnit()
        assert n._phi(-1.0e6) == 0.0

    def test_phi_rejects_non_finite_synaptic_current(self):
        n = WongWangUnit()
        with pytest.raises(ValueError, match="synaptic current"):
            n._phi(np.nan)

    def test_phi_rejects_overflowed_transfer_response(self):
        n = WongWangUnit()
        with pytest.raises(FloatingPointError, match="transfer response"):
            n._phi(sys.float_info.max)

    def test_derivative_guard_rejects_non_finite_stage(self):
        n = WongWangUnit()
        n.j_cross = 0.0
        with pytest.raises(FloatingPointError, match="derivative"):
            n._derivatives(-sys.float_info.max, 0.1, 0.0, 0.0, 0.0, 0.0)

    def test_rejects_non_finite_rng_sample_before_mutation(self, monkeypatch: pytest.MonkeyPatch):
        n = WongWangUnit()
        before = (n.s1, n.s2)
        monkeypatch.setattr(np.random, "randn", lambda: np.inf)
        with pytest.raises(FloatingPointError, match="noise sample"):
            n.step(0.1, 0.0)
        assert (n.s1, n.s2) == before

    def test_rejects_non_finite_rk4_candidate_before_mutation(self, monkeypatch: pytest.MonkeyPatch):
        n = WongWangUnit()
        n.dt = sys.float_info.max
        before = (n.s1, n.s2)

        def huge_derivative(*_args: float) -> tuple[float, float, float, float]:
            return sys.float_info.max, sys.float_info.max, 0.0, 0.0

        monkeypatch.setattr(n, "_derivatives", huge_derivative)
        with pytest.raises(FloatingPointError, match="candidate state"):
            n.step(0.1, 0.0)
        assert (n.s1, n.s2) == before

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
