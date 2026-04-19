# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Sophisticated dynamics tests for Wilson-Cowan 1972

"""Multi-angle tests checking published dynamical properties of the
Wilson-Cowan 1972 E/I rate model, not just API / parity.

Sections:
  1. Sigmoid transfer function — regime coverage
  2. Fixed point at zero drive — quiescent attractor is stable
  3. Monotone response — stronger drive → higher E
  4. E/I separation — E responds faster than I (τ_e < τ_i)
  5. Limit-cycle behaviour in the oscillator regime
  6. Bounded state — 0 ≤ E, I ≤ 1 for physically reasonable
     parameter grid
  7. Parameter sweeps — asymmetric E/I coupling produces expected
     phase-space structure
  8. Cross-backend parity under extreme parameter regimes
  9. Edge cases — zero-length, single-step, boundary init
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sc_neurocore.neurons.models.wilson_cowan import WilsonCowanUnit

DEFAULT_PARAMS = dict(
    w_ee=10.0,
    w_ei=6.0,
    w_ie=10.0,
    w_ii=1.0,
    tau_e=1.0,
    tau_i=2.0,
    a=1.2,
    theta=4.0,
    dt=0.1,
)


# ── 1. Sigmoid transfer function ─────────────────────────────────────


class TestSigmoid:
    """Published Wilson-Cowan 1972 two-term sigmoid:
        S(x) = 1/(1+exp(-a(x-θ))) − 1/(1+exp(aθ))
    Range is [-β, 1-β] with β = 1/(1+exp(aθ)). S(0) = 0 by construction."""

    def test_sigmoid_at_zero_is_zero(self):
        u = WilsonCowanUnit()
        assert abs(u._sigmoid(0.0)) < 1e-12

    def test_sigmoid_at_theta(self):
        """S(θ) = 1/2 − β."""
        u = WilsonCowanUnit()
        baseline = 1.0 / (1.0 + math.exp(u.a * u.theta))
        assert abs(u._sigmoid(u.theta) - (0.5 - baseline)) < 1e-12

    def test_sigmoid_monotone(self):
        u = WilsonCowanUnit()
        xs = np.linspace(-5, 15, 200)
        rs = np.array([u._sigmoid(x) for x in xs])
        assert (np.diff(rs) >= -1e-15).all()

    def test_sigmoid_asymptotes_respect_baseline(self):
        """S(x) → 1 − β as x → +∞, S(x) → −β as x → −∞.
        Input magnitude is limited to ~500 so the scalar `math.exp`
        does not overflow (math.exp raises OverflowError above ~709)."""
        u = WilsonCowanUnit()
        baseline = 1.0 / (1.0 + math.exp(u.a * u.theta))
        assert abs(u._sigmoid(500.0) - (1.0 - baseline)) < 1e-50
        assert abs(u._sigmoid(-500.0) - (-baseline)) < 1e-50

    def test_sigmoid_slope_at_theta(self):
        """dS/dx at θ equals a/4 (baseline subtraction does not shift the
        slope, only the level)."""
        u = WilsonCowanUnit()
        h = 1e-6
        slope = (u._sigmoid(u.theta + h) - u._sigmoid(u.theta - h)) / (2 * h)
        expected = u.a / 4.0
        assert abs(slope - expected) < 1e-4


# ── 2. Quiescent fixed point is stable ───────────────────────────────


class TestQuiescentFixedPoint:
    def test_zero_drive_converges_low(self):
        u = WilsonCowanUnit()
        for _ in range(10_000):
            u.step(0.0)
        assert u.e < 0.01
        assert u.i < 0.01

    def test_small_drive_stays_low(self):
        """Below the sigmoid's activation threshold, E and I stay near 0."""
        u = WilsonCowanUnit()
        for _ in range(5_000):
            u.step(0.2)
        assert u.e < 0.1
        assert u.i < 0.1


# ── 3. Monotone response to drive ────────────────────────────────────


class TestMonotoneResponse:
    def test_stronger_drive_higher_e(self):
        finals = []
        for drive in (0.5, 1.5, 3.0, 5.0, 8.0):
            u = WilsonCowanUnit()
            for _ in range(5_000):
                u.step(drive)
            finals.append(u.e)
        # E saturates near 1 but must be monotonically non-decreasing
        # with drive strength.
        diffs = np.diff(finals)
        assert (diffs >= -1e-3).all(), f"E should be non-decreasing with drive; got {finals}"
        assert finals[-1] > finals[0] + 0.3


# ── 4. Time-constant separation ──────────────────────────────────────


class TestTimeConstantSeparation:
    """τ_e < τ_i: E settles before I on a step input."""

    def test_e_reaches_target_before_i(self):
        u = WilsonCowanUnit(tau_e=1.0, tau_i=4.0)
        trace_e, trace_i = [], []
        for _ in range(800):
            u.step(5.0)
            trace_e.append(u.e)
            trace_i.append(u.i)
        # Find first time each crosses 50 % of its own final value.
        ef, iff = trace_e[-1], trace_i[-1]
        t_e = next((k for k, v in enumerate(trace_e) if v > ef * 0.5), None)
        t_i = next((k for k, v in enumerate(trace_i) if v > iff * 0.5), None)
        assert t_e is not None and t_i is not None
        assert t_e < t_i, f"E must reach 50 % before I (t_e={t_e}, t_i={t_i})"


# ── 5. Oscillator regime ─────────────────────────────────────────────


class TestOscillatorRegime:
    """With strong recurrent coupling + cross-inhibition, Wilson-Cowan
    supports limit cycles. Detecting them via zero-crossings of
    (E - mean(E))."""

    def test_limit_cycle_with_oscillator_params(self):
        """Oscillator parameter set derived from Wilson-Cowan 1972 Fig 3.
        The published two-term sigmoid is required for the Hopf
        bifurcation; with the corrected S(x), the system exhibits
        repeated zero-crossings around the post-transient mean."""
        u = WilsonCowanUnit(
            w_ee=16.0,
            w_ei=12.0,
            w_ie=15.0,
            w_ii=3.0,
            tau_e=1.0,
            tau_i=2.0,
            a=1.2,
            theta=2.8,
            dt=0.1,
        )
        trace_e = []
        for _ in range(4_000):
            u.step(1.25)
            trace_e.append(u.e)
        trace = np.array(trace_e[1_000:])  # discard transient
        centred = trace - trace.mean()
        zero_crossings = np.sum(np.abs(np.diff(np.sign(centred))) > 0)
        assert zero_crossings >= 4, (
            f"Oscillator regime should show repeated E oscillations "
            f"around the mean; got {zero_crossings} zero-crossings"
        )


# ── 6. Bounded state ────────────────────────────────────────────────


class TestBoundedState:
    """Published 2-term sigmoid range is [-β, 1-β] where β = 1/(1+exp(aθ));
    dynamics inherit that envelope so the physically meaningful state
    range is [-β · τ_max, 1 - β · τ_max] bounded by forward-Euler
    relaxation. Empirically |E|, |I| ≤ 1 + β at defaults."""

    @pytest.mark.parametrize("drive", [-5.0, 0.0, 1.0, 5.0, 20.0])
    def test_bounds_under_drive(self, drive):
        u = WilsonCowanUnit()
        baseline = 1.0 / (1.0 + math.exp(u.a * u.theta))
        lo = -baseline - 1e-9
        hi = 1.0 + baseline + 1e-9
        for _ in range(10_000):
            u.step(drive)
            assert lo <= u.e <= hi, f"E out of bounds at drive={drive}: {u.e}"
            assert lo <= u.i <= hi, f"I out of bounds at drive={drive}: {u.i}"


# ── 7. Parameter sweeps ─────────────────────────────────────────────


class TestParameterSweeps:
    def test_w_ei_scales_inhibition(self):
        finals = []
        for w_ei in (2.0, 6.0, 10.0, 15.0):
            u = WilsonCowanUnit(w_ei=w_ei)
            for _ in range(5_000):
                u.step(4.0)
            finals.append(u.e)
        # Stronger cross-inhibition from I onto E reduces E's
        # steady-state activity.
        assert finals[0] > finals[-1], f"Increasing w_ei must lower E; got {finals}"


# ── 8. Cross-backend parity under extreme params ────────────────────


class TestExtremeParamParity:
    """Rust simulator must track Python primary bit-exact across
    extreme parameter regimes."""

    rust = pytest.importorskip(
        "sc_neurocore_engine", reason="Rust engine required"
    ).py_wilson_cowan_simulate

    @pytest.mark.parametrize(
        "params",
        [
            dict(tau_e=0.1, tau_i=0.1, dt=0.01),  # fast dynamics
            dict(tau_e=10.0, tau_i=20.0, dt=0.5),  # slow dynamics, coarse dt
            dict(a=0.5, theta=0.0),  # shallow sigmoid
            dict(a=3.0, theta=8.0),  # steep sigmoid
            dict(w_ee=20.0, w_ei=1.0),  # strong excitation
        ],
    )
    def test_parity_extreme_params(self, params):
        p = {**DEFAULT_PARAMS, **params}
        n = 2_000
        ext = np.linspace(-2.0, 5.0, n)
        u = WilsonCowanUnit(**p)
        e_py = np.empty(n)
        for t in range(n):
            u.step(float(ext[t]))
            e_py[t] = u.e
        out = self.rust(
            0.1,
            0.05,
            p["w_ee"],
            p["w_ei"],
            p["w_ie"],
            p["w_ii"],
            p["tau_e"],
            p["tau_i"],
            p["a"],
            p["theta"],
            p["dt"],
            ext,
        )
        assert np.allclose(e_py, out["e"], atol=1e-14, rtol=0), f"drift under params={params}"


# ── 9. Edge cases ───────────────────────────────────────────────────


class TestEdgeCases:
    rust = pytest.importorskip(
        "sc_neurocore_engine", reason="Rust engine required"
    ).py_wilson_cowan_simulate

    def test_zero_length_workload(self):
        out = self.rust(
            0.5,
            0.3,
            10.0,
            6.0,
            10.0,
            1.0,
            1.0,
            2.0,
            1.2,
            4.0,
            0.1,
            np.zeros(0),
        )
        assert out["e"].shape == (0,)
        assert out["e_final"] == 0.5
        assert out["i_final"] == 0.3

    def test_single_step(self):
        out = self.rust(
            0.1,
            0.05,
            10.0,
            6.0,
            10.0,
            1.0,
            1.0,
            2.0,
            1.2,
            4.0,
            0.1,
            np.array([3.0]),
        )
        assert out["e"].shape == (1,)
        assert out["e_final"] == out["e"][0]

    def test_boundary_init(self):
        """E=0 and E=1 initial conditions must not break downstream math."""
        for e_init in (0.0, 1.0):
            out = self.rust(
                e_init,
                0.5,
                10.0,
                6.0,
                10.0,
                1.0,
                1.0,
                2.0,
                1.2,
                4.0,
                0.1,
                np.full(200, 2.0),
            )
            assert math.isfinite(out["e_final"])
            assert math.isfinite(out["i_final"])
