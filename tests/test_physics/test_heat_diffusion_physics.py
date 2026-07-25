# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Feynman-Kac heat diffusion physics contracts

from __future__ import annotations


import numpy as np

from sc_neurocore.physics.heat import FeynmanKacHeatSolver


def test_diffusivity_drives_spread() -> None:
    """Higher α MUST produce wider spread at the same final time.

    Variance of a free Brownian motion at time T is `2αT`.
    A solver that ignores α (the Antigravity bug) would give
    the same spread for any α — this test catches that regression.
    """
    final_t = 0.05  # short time so reflection is negligible
    n_steps = 50

    s_lo = FeynmanKacHeatSolver(
        length=10.0,
        diffusivity=0.1,
        num_walkers=10_000,
        dt=final_t / n_steps,
        seed=42,
    )
    s_hi = FeynmanKacHeatSolver(
        length=10.0,
        diffusivity=1.0,
        num_walkers=10_000,
        dt=final_t / n_steps,
        seed=42,
    )
    # Start both at the centre to avoid early-reflection bias
    s_lo.set_initial_delta(5.0)
    s_hi.set_initial_delta(5.0)
    s_lo.step(n_steps)
    s_hi.step(n_steps)

    var_lo = float(np.var(s_lo.walkers))
    var_hi = float(np.var(s_hi.walkers))

    # Theory: var_hi / var_lo ≈ 1.0 / 0.1 = 10.0
    ratio = var_hi / var_lo
    assert 7.0 < ratio < 13.0, (
        f"diffusivity not driving spread: var_lo={var_lo:.4f}, "
        f"var_hi={var_hi:.4f}, ratio={ratio:.2f}, expected ≈ 10"
    )


def test_free_brownian_variance_matches_analytic() -> None:
    """E[X_T²] = 2αT for free Brownian motion (no boundary effects)."""
    alpha = 0.5
    final_t = 0.01  # so 2αT = 0.01 ≪ L²/4 = 25
    n_steps = 100
    s = FeynmanKacHeatSolver(
        length=10.0,
        diffusivity=alpha,
        num_walkers=20_000,
        dt=final_t / n_steps,
        seed=99,
    )
    s.set_initial_delta(5.0)  # start at centre
    s.step(n_steps)

    # Variance of the position about the start
    centred = s.walkers - 5.0
    measured_var = float(np.mean(centred**2))
    expected_var = 2.0 * alpha * final_t  # = 0.01

    rel_err = abs(measured_var - expected_var) / expected_var
    assert rel_err < 0.05, (
        f"Brownian variance off: measured={measured_var:.6f}, "
        f"expected={expected_var:.6f}, rel_err={rel_err:.4f}"
    )


def test_long_time_converges_to_uniform_on_reflective_domain() -> None:
    """For pure diffusion with reflective BC, density → 1/L as t → ∞.

    Test that after enough mixing time, the histogram is close to
    uniform (KS-like stat: max bin / mean bin is bounded).
    """
    L = 1.0
    s = FeynmanKacHeatSolver(
        length=L,
        diffusivity=1.0,
        num_walkers=20_000,
        dt=1e-3,
        seed=11,
    )
    s.set_initial_delta(0.5)
    s.step(2_000)  # 2 sec; mixing time ~ L²/π²α = 0.1 s

    density = s.get_density(n_bins=20)
    target = 1.0 / L  # uniform density
    rel_err = float(np.max(np.abs(density - target)) / target)
    assert rel_err < 0.10, (
        f"density not uniform after long time: max rel-err {rel_err:.4f}, density={density}"
    )


def test_expectation_matches_initial_value_at_t_zero() -> None:
    """At t=0, E[f(X_0)] = f(x_0) for any observable f."""
    s = FeynmanKacHeatSolver(num_walkers=1000, seed=3)
    s.set_initial_delta(0.42)
    # f(x) = x² → E[X_0²] = 0.42² = 0.1764
    e = s.expectation(lambda x: x**2)
    assert abs(e - 0.42**2) < 1e-9
