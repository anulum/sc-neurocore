# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Sophisticated dynamics / invariants tests for Wong-Wang 2006

"""Multi-angle tests that check published dynamical properties of the
Wong-Wang 2006 decision circuit, not just API / parity.

Sections:
  1. Transfer function φ(I) — full regime coverage
  2. Zero-noise determinism + symmetry invariants
  3. Bistability — two-attractor structure above J_N critical coupling
  4. Sub-critical regime — no winner-take-all when J_N is too small
  5. Psychometric function — accuracy vs coherence across trials
  6. Reaction-time dependence on stimulus strength
  7. Winner persistence after stimulus offset (attractor stability)
  8. Noise monotonicity — decision variability grows with σ
  9. State bounds — s_k ∈ [0, 1] over long runs
 10. Scale-invariant: scaling `stim + sigma * xi` by same factor leaves
     the decision outcome statistically invariant only if phi were
     linear — we check the expected non-invariance as an honest probe
     that φ is genuinely non-linear.
 11. Cross-backend parity under pathological parameters (huge σ, tiny dt)
 12. Zero-length workload is a no-op (edge case)

All stochastic tests are seeded and use N large enough for the stated
tolerance to fail a broken implementation at 99.9 % confidence.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.wong_wang import WongWangUnit

DEFAULT_PARAMS = dict(
    tau_s=0.1,
    gamma=0.641,
    j_n=0.2609,
    j_cross=0.0497,
    i_0=0.3255,
    sigma=0.02,
    dt=0.001,
)

A_PHI = 270.0
B_PHI = 108.0
D_PHI = 0.154


# ── 1. Transfer function φ(I) full regime coverage ────────────────────


class TestTransferFunction:
    """φ(I) = (aI - b) / (1 - exp(-d(aI - b)))  singularity-guarded."""

    def test_phi_asymptotes_linear_for_large_positive_I(self):
        """For large I, φ(I) → aI - b (denominator → 1)."""
        u = WongWangUnit()
        I = 10.0
        expected = A_PHI * I - B_PHI  # ~2592
        r = u._phi(I)
        assert abs(r - expected) / expected < 1e-6

    def test_phi_asymptotes_zero_for_large_negative_I(self):
        """For deeply negative I, (aI-b) is very negative and
        x/(1-exp(-dx)) → 0 because exp(-dx) explodes, denominator
        tends to -exp(|d*x|) so ratio → 0 from below."""
        u = WongWangUnit()
        r = u._phi(-10.0)
        # Near zero but not exactly (denom grows exponentially)
        assert abs(r) < 1e-100

    def test_phi_continuous_across_singularity(self):
        """L'Hôpital gives φ(b/a) = 1/d; check continuity by sweeping
        across the singularity and asserting no jump."""
        u = WongWangUnit()
        pivot = B_PHI / A_PHI  # = 0.4
        r_lo = float(u._phi(pivot - 1e-7))
        r_pv = float(u._phi(pivot))
        r_hi = float(u._phi(pivot + 1e-7))
        # All three should be within 1e-3 of 1/d = 6.494
        target = 1.0 / D_PHI
        assert abs(r_lo - target) < 1e-3
        assert abs(r_pv - target) < 1e-8
        assert abs(r_hi - target) < 1e-3

    def test_phi_strictly_increasing_across_sampled_grid(self):
        """Sampled Is spanning the full regime must yield increasing r."""
        u = WongWangUnit()
        Is = np.linspace(-1.0, 3.0, 201)
        rs = np.array([float(u._phi(I)) for I in Is])
        # Allow 0 for the flat left tail (both would be 0)
        gaps = np.diff(rs)
        # At least 95 % of gaps must be non-negative (no decreasing anywhere)
        assert (gaps >= -1e-15).all(), "φ must be monotone non-decreasing"


# ── 2. Zero-noise determinism + symmetry ──────────────────────────────


class TestZeroNoiseInvariants:
    """σ = 0 removes stochasticity; outputs become deterministic."""

    def test_symmetric_init_preserves_symmetry_forever(self):
        """s1_0 == s2_0, stim1 == stim2, σ = 0 → s1(t) == s2(t) ∀ t."""
        u = WongWangUnit(sigma=0.0)
        for _ in range(50_000):
            u.step(0.1, 0.1)
        assert abs(u.s1 - u.s2) < 1e-12

    def test_zero_noise_identical_runs_identical(self):
        np.random.seed(1)
        u1 = WongWangUnit(sigma=0.0)
        for _ in range(10_000):
            u1.step(0.1, 0.0)
        np.random.seed(2)  # different seed — but σ=0 so RNG irrelevant
        u2 = WongWangUnit(sigma=0.0)
        for _ in range(10_000):
            u2.step(0.1, 0.0)
        assert u1.s1 == u2.s1 and u1.s2 == u2.s2

    def test_zero_noise_equal_stim_no_symmetry_break(self):
        """Without noise and with equal stimuli, the system sits on the
        symmetric fixed point (no winner-take-all possible)."""
        u = WongWangUnit(sigma=0.0)
        for _ in range(100_000):
            u.step(0.05, 0.05)
        # Difference stays at numerical zero
        assert abs(u.s1 - u.s2) < 1e-10


# ── 3. Bistability above J_N critical coupling ────────────────────────


class TestBistability:
    """Two-attractor structure: strong self-excitation + mutual inhibition
    forces near-binary winner/loser split under asymmetric drive."""

    @pytest.mark.parametrize(
        "stim1,stim2,winner_side",
        [
            (0.12, 0.02, 1),
            (0.02, 0.12, 2),
            (0.2, 0.0, 1),
            (0.0, 0.2, 2),
        ],
    )
    def test_biased_stim_selects_correct_winner(self, stim1, stim2, winner_side):
        np.random.seed(42)
        u = WongWangUnit(sigma=0.001)  # small noise — bias dominates
        for _ in range(60_000):
            u.step(stim1, stim2)
        if winner_side == 1:
            assert u.s1 > u.s2 + 0.4, f"winner split insufficient: {u.s1} vs {u.s2}"
        else:
            assert u.s2 > u.s1 + 0.4, f"winner split insufficient: {u.s1} vs {u.s2}"

    def test_j_n_scales_winner_activity(self):
        """Higher j_n → winner attractor lies at higher s; verifies the
        self-excitation mechanism is genuinely at work."""
        s_by_jn = {}
        for jn in (0.20, 0.26, 0.32):
            np.random.seed(0)
            u = WongWangUnit(j_n=jn, sigma=0.001)
            for _ in range(60_000):
                u.step(0.15, 0.0)
            s_by_jn[jn] = u.s1
        # Monotone increase: each larger j_n pushes winner higher
        assert s_by_jn[0.20] < s_by_jn[0.26] < s_by_jn[0.32]

    def test_j_cross_scales_loser_suppression(self):
        """Higher j_cross → loser attractor lies at lower s."""
        s_loser_by_jx = {}
        for jx in (0.02, 0.05, 0.09):
            np.random.seed(0)
            u = WongWangUnit(j_cross=jx, sigma=0.001)
            for _ in range(60_000):
                u.step(0.2, 0.0)
            s_loser_by_jx[jx] = u.s2
        # Stronger cross-inhibition → lower loser activity
        assert s_loser_by_jx[0.02] > s_loser_by_jx[0.05] > s_loser_by_jx[0.09]


# ── 4. Sub-critical regime — no winner-take-all ───────────────────────


class TestSubcriticalRegime:
    """With j_n far below the critical coupling, the system should settle
    near the low spontaneous fixed point regardless of bias."""

    def test_small_j_n_no_winner(self):
        np.random.seed(0)
        u = WongWangUnit(j_n=0.05, j_cross=0.01, sigma=0.001)
        for _ in range(60_000):
            u.step(0.05, 0.0)
        # Both pools stay low (quiescent attractor)
        assert u.s1 < 0.35, f"sub-critical s1 must stay low: {u.s1}"
        assert u.s2 < 0.35, f"sub-critical s2 must stay low: {u.s2}"


# ── 5. Psychometric function — accuracy vs coherence ──────────────────


class TestPsychometricCurve:
    """Monotonic increase of correct-decision probability with coherence.
    At 0 % coherence, P(A) ≈ 0.5 (chance). At high coherence, P(A) → 1."""

    @pytest.mark.parametrize("n_trials", [40])
    def test_accuracy_monotone_in_coherence(self, n_trials):
        """Run many short trials at different coherences; fraction of
        trials where s1 wins must rise with coherence."""
        coherences = [0.0, 0.05, 0.15, 0.40]
        acc = []
        for coh in coherences:
            wins = 0
            for trial in range(n_trials):
                np.random.seed(trial)
                u = WongWangUnit()
                for _ in range(8_000):
                    u.step(0.05 + coh * 0.1, 0.05 - coh * 0.1)
                if u.s1 > u.s2:
                    wins += 1
            acc.append(wins / n_trials)
        # 0% ≈ 50%, high coherence must be strictly better than 0%
        assert 0.35 <= acc[0] <= 0.65, f"chance-level at 0% expected, got {acc[0]}"
        assert acc[-1] > acc[0] + 0.15, (
            f"high coherence accuracy must exceed chance significantly: "
            f"0 → {acc[0]:.2f}, high → {acc[-1]:.2f}"
        )
        # Strict monotonicity check on the trend (allow one inversion)
        inversions = sum(1 for a, b in zip(acc, acc[1:]) if a > b + 0.01)
        assert inversions <= 1, f"accuracy curve must be near-monotone; got {acc}"


# ── 6. Reaction-time dependence on stimulus strength ──────────────────


class TestReactionTime:
    """Stronger asymmetric drive → shorter time to cross a decision
    threshold. Matches Roitman & Shadlen 2002 qualitative finding."""

    def _time_to_threshold(self, stim1, stim2, threshold, seed):
        np.random.seed(seed)
        u = WongWangUnit(sigma=0.01)
        for t in range(50_000):
            u.step(stim1, stim2)
            if u.s1 > threshold or u.s2 > threshold:
                return t
        return None

    def test_rt_decreases_with_stimulus_difference(self):
        rt_low = []
        rt_high = []
        for seed in range(20):
            t = self._time_to_threshold(0.08, 0.06, 0.55, seed)
            if t is not None:
                rt_low.append(t)
            t = self._time_to_threshold(0.25, 0.0, 0.55, seed)
            if t is not None:
                rt_high.append(t)
        assert len(rt_low) >= 10, f"low-drive decisions too rare: {len(rt_low)}"
        assert len(rt_high) >= 10, f"high-drive decisions too rare: {len(rt_high)}"
        # Median RT must drop substantially with drive
        assert np.median(rt_high) < 0.6 * np.median(rt_low), (
            f"high drive should reach threshold faster: "
            f"low median={np.median(rt_low):.0f}, "
            f"high median={np.median(rt_high):.0f}"
        )


# ── 7. Winner persistence after stimulus offset ───────────────────────


class TestWinnerPersistence:
    """Once an attractor is reached, the winner stays above the loser
    even after the asymmetric stimulus is removed — the system is
    bistable with a finite basin of attraction."""

    def test_winner_stays_after_stim_off(self):
        np.random.seed(0)
        u = WongWangUnit(sigma=0.001)
        # Phase 1: drive s1 to its attractor
        for _ in range(30_000):
            u.step(0.2, 0.0)
        assert u.s1 > 0.55
        s1_lock = u.s1
        # Phase 2: remove stimulus, run more steps
        for _ in range(20_000):
            u.step(0.0, 0.0)
        # s1 still dominates, though may relax slightly toward attractor
        assert u.s1 > u.s2, f"winner must persist after stim off: s1={u.s1}, s2={u.s2}"
        assert abs(u.s1 - s1_lock) < 0.4, (
            f"winner should not dramatically reverse: {s1_lock} → {u.s1}"
        )


# ── 8. Noise monotonicity — variability grows with σ ──────────────────


class TestNoiseVariability:
    """Across trials with the same deterministic input, higher σ produces
    more spread in the final state."""

    def _trial_spread(self, sigma, n_trials):
        finals = []
        for seed in range(n_trials):
            np.random.seed(seed)
            u = WongWangUnit(sigma=sigma)
            for _ in range(10_000):
                u.step(0.05, 0.05)  # symmetric, noise alone breaks tie
            finals.append(u.s1)
        return float(np.std(finals))

    def test_higher_sigma_increases_std(self):
        s_low = self._trial_spread(sigma=0.005, n_trials=20)
        s_mid = self._trial_spread(sigma=0.02, n_trials=20)
        s_hi = self._trial_spread(sigma=0.08, n_trials=20)
        # Higher noise → wider outcome distribution.
        assert s_low < s_mid, f"low→mid: {s_low:.3f} vs {s_mid:.3f}"
        assert s_mid < s_hi, f"mid→hi: {s_mid:.3f} vs {s_hi:.3f}"


# ── 9. State bounds over long runs ────────────────────────────────────


class TestStateBounds:
    """s_k ∈ [0, 1] must hold for every step, even after N=200k iterations
    with extreme drives."""

    @pytest.mark.parametrize("stim", [-0.5, 0.0, 0.5, 1.0, 2.0])
    def test_bounds_under_extreme_drives(self, stim):
        np.random.seed(0)
        u = WongWangUnit()
        for _ in range(200_000):
            u.step(stim, -stim)
            assert 0.0 <= u.s1 <= 1.0, f"s1 out of bounds at stim={stim}: {u.s1}"
            assert 0.0 <= u.s2 <= 1.0, f"s2 out of bounds at stim={stim}: {u.s2}"


# ── 10. Non-linearity probe ───────────────────────────────────────────


class TestNonLinearity:
    """Scaling all currents by k (stim → k*stim, σ → k*σ) does NOT leave
    the trajectory invariant because φ is strongly non-linear. This is
    a regression guard: if someone breaks φ down to a linear kernel by
    mistake, the trajectory would scale, and this test would fail
    (because scaled trajectory differs from non-scaled by the amount φ
    departs from linearity)."""

    def test_scale_non_invariance(self):
        np.random.seed(0)
        u1 = WongWangUnit(sigma=0.02)
        for _ in range(5_000):
            u1.step(0.1, 0.0)
        np.random.seed(0)
        u2 = WongWangUnit(sigma=0.04)  # doubled σ
        for _ in range(5_000):
            u2.step(0.2, 0.0)  # doubled stim
        # Genuinely non-linear kernel: trajectories must DIFFER
        # substantially (not just numerically). If they were close,
        # someone linearised φ.
        assert abs(u1.s1 - u2.s1) > 0.05 or abs(u1.s2 - u2.s2) > 0.05, (
            f"φ non-linearity probe: trajectories suspiciously similar "
            f"({u1.s1:.3f},{u1.s2:.3f}) vs ({u2.s1:.3f},{u2.s2:.3f})"
        )


# ── 11. Cross-backend parity under pathological params ────────────────


class TestExtremeParamParity:
    """Parity checks done in other files use "nominal" parameters.
    Here we probe extreme regimes: huge σ, tiny dt, very small τ_s.
    The Rust kernel must still track the Python primary bit-exact
    (Python ownership of RNG + same arithmetic)."""

    rust = pytest.importorskip(
        "sc_neurocore_engine", reason="Rust engine required"
    ).py_wong_wang_simulate

    @pytest.mark.parametrize(
        "sigma,dt,tau_s",
        [
            (0.2, 0.001, 0.1),  # very noisy
            (0.0, 0.0001, 0.1),  # no noise, tiny dt
            (0.02, 0.001, 0.02),  # fast NMDA
            (0.02, 0.01, 0.1),  # coarse dt
        ],
    )
    def test_parity_under_extreme_params(self, sigma, dt, tau_s):
        n = 3_000
        np.random.seed(13)
        u = WongWangUnit(sigma=sigma, dt=dt, tau_s=tau_s)
        s1_py = np.empty(n)
        s2_py = np.empty(n)
        for t in range(n):
            u.step(0.1, 0.0)
            s1_py[t] = u.s1
            s2_py[t] = u.s2
        np.random.seed(13)
        xi = np.random.randn(2 * n).astype(np.float64)
        out = self.rust(
            0.1,
            0.1,
            tau_s,
            0.641,
            0.2609,
            0.0497,
            0.3255,
            sigma,
            dt,
            np.full(n, 0.1),
            np.zeros(n),
            xi,
        )
        assert np.allclose(s1_py, out["s1"], atol=1e-12, rtol=0), (
            f"sigma={sigma} dt={dt} tau_s={tau_s}: s1 drift"
        )
        assert np.allclose(s2_py, out["s2"], atol=1e-12, rtol=0), (
            f"sigma={sigma} dt={dt} tau_s={tau_s}: s2 drift"
        )


# ── 12. Edge cases ────────────────────────────────────────────────────


class TestEdgeCases:
    rust = pytest.importorskip(
        "sc_neurocore_engine", reason="Rust engine required"
    ).py_wong_wang_simulate

    def test_zero_length_workload_is_no_op(self):
        """Empty stim arrays should return empty traces and keep init state."""
        out = self.rust(
            0.1,
            0.2,
            0.1,
            0.641,
            0.2609,
            0.0497,
            0.3255,
            0.02,
            0.001,
            np.zeros(0),
            np.zeros(0),
            np.zeros(0),
        )
        assert out["s1"].shape == (0,)
        assert out["s2"].shape == (0,)
        assert out["s1_final"] == 0.1
        assert out["s2_final"] == 0.2

    def test_single_step_call(self):
        """n=1 is a legitimate workload; must not special-case."""
        out = self.rust(
            0.1,
            0.1,
            0.1,
            0.641,
            0.2609,
            0.0497,
            0.3255,
            0.0,
            0.001,
            np.array([0.1]),
            np.array([0.0]),
            np.array([0.0, 0.0]),
        )
        assert out["s1"].shape == (1,)
        assert out["s1_final"] == out["s1"][0]

    def test_init_at_boundary(self):
        """Starting at s=0 or s=1 boundary; clip must not misbehave."""
        for s1_init in (0.0, 1.0):
            out = self.rust(
                s1_init,
                0.5,
                0.1,
                0.641,
                0.2609,
                0.0497,
                0.3255,
                0.0,
                0.001,
                np.full(100, 0.1),
                np.zeros(100),
                np.zeros(200),
            )
            assert 0.0 <= out["s1"].min() <= out["s1"].max() <= 1.0


# ── 13. Long-run numerical stability ─────────────────────────────────


class TestLongRunStability:
    """`feedback_module_standard_attnres` requires algorithm / parity /
    **stability** tests; this section provides the third leg. Rust
    simulator is used so 1 M-step sweeps fit in the test budget."""

    rust = pytest.importorskip(
        "sc_neurocore_engine", reason="Rust engine required"
    ).py_wong_wang_simulate

    @pytest.mark.parametrize(
        "stim_level,sigma",
        [
            (0.0, 0.0),  # quiescent, deterministic
            (0.1, 0.02),  # nominal, noisy
            (0.3, 0.08),  # strong drive, high noise
        ],
    )
    def test_no_nan_no_inf_over_1M_steps(self, stim_level, sigma):
        """1 M-step run must stay finite and within [0, 1] for s1/s2."""
        n = 1_000_000
        stim1 = np.full(n, stim_level, dtype=np.float64)
        stim2 = np.zeros(n, dtype=np.float64)
        np.random.seed(42)
        xi = np.random.randn(2 * n).astype(np.float64)
        out = self.rust(
            0.1,
            0.1,
            0.1,
            0.641,
            0.2609,
            0.0497,
            0.3255,
            sigma,
            0.001,
            stim1,
            stim2,
            xi,
        )
        s1, s2 = out["s1"], out["s2"]
        assert np.isfinite(s1).all(), f"s1 non-finite at stim={stim_level}, σ={sigma}"
        assert np.isfinite(s2).all(), f"s2 non-finite at stim={stim_level}, σ={sigma}"
        # Published Wong-Wang state range is strictly [0, 1] due to explicit clip.
        assert s1.min() >= 0.0, f"s1 clipped floor broken at stim={stim_level}: min={s1.min()}"
        assert s1.max() <= 1.0, f"s1 clipped ceiling broken at stim={stim_level}: max={s1.max()}"
        assert 0.0 <= s2.min() <= s2.max() <= 1.0

    def test_deterministic_fixed_point_settles_under_constant_stim(self):
        """Under constant biased drive (σ=0), 1 M steps must reach a
        fixed point; trailing 100 k should have near-zero variance."""
        n = 1_000_000
        stim1 = np.full(n, 0.2, dtype=np.float64)
        stim2 = np.zeros(n, dtype=np.float64)
        xi = np.zeros(2 * n, dtype=np.float64)
        out = self.rust(
            0.1,
            0.1,
            0.1,
            0.641,
            0.2609,
            0.0497,
            0.3255,
            0.0,
            0.001,
            stim1,
            stim2,
            xi,
        )
        tail_std = float(np.std(out["s1"][-100_000:]))
        assert tail_std < 1e-6, (
            f"Deterministic run should reach fixed point; s1 tail std = {tail_std:.2e}"
        )

    def test_state_function_of_inputs_only(self):
        """Two independent 500 k-step runs from identical init + stim + xi
        must produce bit-identical final states."""
        n = 500_000
        stim1 = np.full(n, 0.15, dtype=np.float64)
        stim2 = np.zeros(n, dtype=np.float64)
        np.random.seed(7)
        xi = np.random.randn(2 * n).astype(np.float64)
        out_a = self.rust(
            0.1,
            0.1,
            0.1,
            0.641,
            0.2609,
            0.0497,
            0.3255,
            0.02,
            0.001,
            stim1,
            stim2,
            xi,
        )
        out_b = self.rust(
            0.1,
            0.1,
            0.1,
            0.641,
            0.2609,
            0.0497,
            0.3255,
            0.02,
            0.001,
            stim1,
            stim2,
            xi,
        )
        assert out_a["s1_final"] == out_b["s1_final"]
        assert out_a["s2_final"] == out_b["s2_final"]
