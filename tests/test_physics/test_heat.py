# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the Feynman-Kac heat solver

"""Physics tests for `sc_neurocore.physics.heat.FeynmanKacHeatSolver`.

The prior heat-solver implementation (discrete lattice walk that ignored the
diffusivity α) was replaced 2026-04-17 with a proper
Brownian-motion-based Feynman-Kac solver. These tests pin the
physics invariants — diffusivity must drive the spread, the
expectation must converge against an analytic reference, mass
must be conserved.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.physics.heat import FeynmanKacHeatSolver, StochasticHeatSolver


def test_alias_points_at_feynman_kac() -> None:
    """The legacy `StochasticHeatSolver` name is now the new class."""
    assert StochasticHeatSolver is FeynmanKacHeatSolver


def test_walkers_must_be_initialised_before_step() -> None:
    s = FeynmanKacHeatSolver(num_walkers=100)
    with pytest.raises(RuntimeError, match="walkers not initialised"):
        s.step()


def test_density_and_expectation_require_initialised_walkers() -> None:
    s = FeynmanKacHeatSolver(num_walkers=100)
    with pytest.raises(RuntimeError, match="walkers not initialised"):
        s.get_density()
    with pytest.raises(RuntimeError, match="walkers not initialised"):
        s.expectation(lambda x: x)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"length": 0.0}, "length"),
        ({"length": float("nan")}, "length"),
        ({"length": True}, "length"),
        ({"diffusivity": -1e-9}, "diffusivity"),
        ({"diffusivity": float("inf")}, "diffusivity"),
        ({"diffusivity": False}, "diffusivity"),
        ({"num_walkers": 0}, "num_walkers"),
        ({"num_walkers": 1.5}, "num_walkers"),
        ({"num_walkers": True}, "num_walkers"),
        ({"dt": 0.0}, "dt"),
        ({"dt": float("nan")}, "dt"),
        ({"dt": False}, "dt"),
        ({"seed": 1.2}, "seed"),
        ({"seed": True}, "seed"),
    ],
)
def test_solver_rejects_nonphysical_configuration(kwargs: dict[str, object], match: str) -> None:
    values = {
        "length": 1.0,
        "diffusivity": 1.0,
        "num_walkers": 10,
        "dt": 1e-3,
        "seed": 42,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        FeynmanKacHeatSolver(**values)


def test_delta_initial_condition_places_all_walkers_at_x0() -> None:
    s = FeynmanKacHeatSolver(length=2.0, num_walkers=500, seed=7)
    s.set_initial_delta(0.5)
    assert np.all(s.walkers == 0.5)
    assert s.time == 0.0


def test_delta_x0_outside_domain_raises() -> None:
    s = FeynmanKacHeatSolver(length=1.0, num_walkers=10)
    with pytest.raises(ValueError, match="outside"):
        s.set_initial_delta(2.0)


def test_density_integrates_to_unity() -> None:
    """Bin counts × bin width should sum to 1 (probability density)."""
    s = FeynmanKacHeatSolver(length=1.0, num_walkers=5_000, seed=1)
    s.set_initial_delta(0.5)
    s.step(50)
    density = s.get_density(n_bins=64)
    integral = density.sum() * (1.0 / 64)
    assert abs(integral - 1.0) < 1e-9


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


def test_reflective_boundaries_keep_walkers_in_domain() -> None:
    """No walker should leak outside [0, L] regardless of step count."""
    s = FeynmanKacHeatSolver(
        length=1.0,
        diffusivity=10.0,
        num_walkers=2_000,
        dt=1e-3,
        seed=2026,
    )
    s.set_initial_delta(0.5)
    s.step(500)
    assert s.walkers.min() >= 0.0
    assert s.walkers.max() <= s.length


def test_exact_reflection_handles_arbitrarily_large_overshoot() -> None:
    x = np.array([-4.2, -0.25, 0.25, 1.25, 4.2])
    folded = FeynmanKacHeatSolver._reflect_into_interval(x, 1.0)
    assert np.allclose(folded, [0.2, 0.25, 0.25, 0.75, 0.2])
    assert np.all((folded >= 0.0) & (folded <= 1.0))


def test_zero_diffusivity_preserves_delta_position() -> None:
    s = FeynmanKacHeatSolver(diffusivity=0.0, num_walkers=128, seed=8)
    s.set_initial_delta(0.37)
    s.step(50)
    assert np.all(s.walkers == 0.37)
    assert s.time == pytest.approx(0.05)


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


def test_set_initial_distribution_zero_density_raises() -> None:
    """All-zero PDF must be rejected."""
    s = FeynmanKacHeatSolver(num_walkers=10)
    with pytest.raises(ValueError, match="must integrate"):
        s.set_initial_distribution(lambda x: np.zeros_like(x))


def test_set_initial_distribution_rejects_invalid_grid_and_density_contract() -> None:
    s = FeynmanKacHeatSolver(num_walkers=10)
    with pytest.raises(ValueError, match="n_grid"):
        s.set_initial_distribution(lambda x: np.ones_like(x), n_grid=0)
    with pytest.raises(ValueError, match="matching x"):
        s.set_initial_distribution(lambda x: np.ones(x.size + 1), n_grid=8)
    with pytest.raises(ValueError, match="finite"):
        s.set_initial_distribution(lambda x: np.full_like(x, np.nan), n_grid=8)


def test_set_initial_distribution_uniform_is_uniform() -> None:
    """A uniform initial PDF should produce a uniform initial histogram."""
    s = FeynmanKacHeatSolver(length=2.0, num_walkers=50_000, seed=22)
    s.set_initial_distribution(lambda x: np.ones_like(x), n_grid=128)
    density = s.get_density(n_bins=20)
    target = 1.0 / 2.0
    rel_err = float(np.max(np.abs(density - target)) / target)
    assert rel_err < 0.05, f"uniform initial sampling off: max rel-err {rel_err:.4f}"


def test_evolve_to_advances_clock() -> None:
    s = FeynmanKacHeatSolver(num_walkers=100, dt=1e-3, seed=4)
    s.set_initial_delta(0.5)
    s.evolve_to(0.05)
    # Allow rounding tolerance from int(round((T-t)/dt))
    assert abs(s.time - 0.05) < 1.5e-3


def test_evolve_to_rejects_nonfinite_time_and_density_bins() -> None:
    s = FeynmanKacHeatSolver(num_walkers=10, seed=5)
    s.set_initial_delta(0.5)
    with pytest.raises(ValueError, match="T"):
        s.evolve_to(float("nan"))
    with pytest.raises(ValueError, match="n_bins"):
        s.get_density(n_bins=0)


def test_evolve_to_backwards_raises() -> None:
    s = FeynmanKacHeatSolver(num_walkers=10, seed=5)
    s.set_initial_delta(0.5)
    s.evolve_to(0.1)
    with pytest.raises(ValueError, match="cannot run backwards"):
        s.evolve_to(0.05)
