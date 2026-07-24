# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHillTononiRK4 from former test_model_hill_tononi.py

"""Focused suite: TestHillTononiRK4 from former test_model_hill_tononi.py."""

from __future__ import annotations

from tests.model_hill_tononi_support import *  # noqa: F403


class TestHillTononiRK4:
    """Guards the candidate-first RK4 integrator and its cross-backend parity.

    The historical forward Euler advanced the four gates from the old voltage and
    then the membrane and intracellular sodium from the freshly updated gates,
    mixing inconsistent states. The production path is RK4 over the six-state
    ``(v, h_na, n_k, m_h, h_t, na_i)`` system with one consistent right-hand side
    per stage; the staggered baseline survives only behind
    ``integrator="baseline_euler"``. The ``I_KNa`` Hill exponent ``3.5`` is
    evaluated as ``b·b·b·sqrt(b)`` so every backend reproduces the trajectory.
    """

    def test_default_integrator_is_rk4(self):
        assert HillTononiNeuron().integrator == "rk4"

    def test_unknown_integrator_rejected(self):
        with pytest.raises(ValueError, match="integrator"):
            HillTononiNeuron(integrator="trapezoid")  # type: ignore[arg-type]

    def test_baseline_euler_path_runs_and_fires(self):
        n = HillTononiNeuron(integrator="baseline_euler")
        assert sum(n.step(0.0) for _ in range(10000)) > 0

    def test_rk4_and_baseline_differ(self):
        rk4 = HillTononiNeuron()
        euler = HillTononiNeuron(integrator="baseline_euler")
        rk4_v = [round(rk4.step(10.0) or rk4.v, 9) for _ in range(500)]
        euler_v = [round(euler.step(10.0) or euler.v, 9) for _ in range(500)]
        assert rk4_v != euler_v

    def test_cross_backend_spike_anchor(self):
        # Pins the Python reference the Rust/Julia/Go/Mojo kernels reproduce
        # bit-for-bit (verified by benchmarks/bench_model_hill_tononi.py).
        n = HillTononiNeuron()
        assert sum(n.step(10.0) for _ in range(200000)) == 694

    def test_sodium_clamped_non_negative(self):
        n = HillTononiNeuron()
        for _ in range(50000):
            n.step(10.0)
            assert n.na_i >= 0.0

    def test_non_finite_current_rejected(self):
        n = HillTononiNeuron()
        with pytest.raises(ValueError, match="current"):
            n.step(math.nan)

    def test_non_finite_state_rejected_on_step(self):
        n = HillTononiNeuron()
        n.v = math.inf
        with pytest.raises(ValueError, match="v"):
            n.step(10.0)

    def test_negative_conductance_rejected(self):
        with pytest.raises(ValueError, match="g_na"):
            HillTononiNeuron(g_na=-1.0)

    def test_non_positive_na_eq_rejected(self):
        with pytest.raises(ValueError, match="na_eq"):
            HillTononiNeuron(na_eq=0.0)

    def test_stays_finite_under_extreme_drive(self):
        # The delayed-rectifier and I_KNa currents grow with V, so the cell
        # self-limits to a high but finite fixed point rather than diverging; the
        # _safe_exp guard keeps the saturating gates finite (Python math.exp would
        # otherwise raise OverflowError where the other backends return +inf).
        n = HillTononiNeuron()
        for _ in range(2000):
            n.step(1e5)
        assert math.isfinite(n.v) and math.isfinite(n.na_i)

    def test_h_current_evolves(self):
        from sc_neurocore.neurons.models.hill_tononi import HillTononiNeuron

        n = HillTononiNeuron()
        for _ in range(100):
            n.step(3.0)
        assert n.m_h != 0.0
