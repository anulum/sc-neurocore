# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real physics routes for the safe alternative-path harness

"""Physics-solver comparison routes for the safe alternative-path harness.

Registers baseline/candidate route pairs for heat diffusion, harmonic
oscillators, and Kuramoto synchronisation, validating fast candidates against
reference solvers under the alternative-path harness.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from sc_neurocore.physics.heat import FeynmanKacHeatSolver
from sc_neurocore.solvers import RK4Solver, StormerVerlet

from .alternative_path import AlternativePathRoute


def _heat_cosine_mode_baseline(
    x_0: float,
    horizon: float,
    mode_index: int = 1,
    *,
    length: float = 1.0,
    diffusivity: float = 0.5,
    num_walkers: int = 40_000,
    dt: float = 1e-4,
    seed: int = 7,
) -> float:
    solver = FeynmanKacHeatSolver(
        length=length,
        diffusivity=diffusivity,
        num_walkers=num_walkers,
        dt=dt,
        seed=seed,
    )
    solver.set_initial_delta(x_0)
    solver.evolve_to(horizon)
    wavenumber = mode_index * math.pi / length
    return solver.expectation(lambda x: np.cos(wavenumber * x))


def _heat_cosine_mode_candidate(
    x_0: float,
    horizon: float,
    mode_index: int = 1,
    *,
    length: float = 1.0,
    diffusivity: float = 0.5,
    num_walkers: int = 40_000,
    dt: float = 1e-4,
    seed: int = 7,
) -> float:
    del num_walkers, dt, seed
    decay = math.exp(-diffusivity * (mode_index * math.pi / length) ** 2 * horizon)
    return decay * math.cos(mode_index * math.pi * x_0 / length)


def make_heat_cosine_mode_route() -> AlternativePathRoute[float]:
    """Route Monte Carlo heat evolution against an exact Neumann cosine mode."""
    return AlternativePathRoute(
        name="physics.heat.cosine-mode",
        baseline=_heat_cosine_mode_baseline,
        candidate=_heat_cosine_mode_candidate,
        summary="Feynman-Kac Monte Carlo vs exact Neumann cosine-mode heat solution",
        expected_behavior=(
            "For cosine-mode initial data, the exact candidate should match the "
            "Monte Carlo baseline within sampling tolerance"
        ),
    )


def _harmonic_oscillator_rhs(_t: float, y: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.array([y[1], -y[0]], dtype=np.float64)


def _harmonic_energy(y: np.ndarray[Any, Any]) -> float:
    return float(0.5 * (y[0] ** 2 + y[1] ** 2))


def _validate_harmonic_inputs(
    q0: float,
    p0: float,
    horizon: float,
    dt: float,
) -> tuple[float, float, float, float]:
    for value, name in ((q0, "q0"), (p0, "p0"), (horizon, "horizon"), (dt, "dt")):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{name} must be a finite number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be a finite number")
    if float(horizon) <= 0.0:
        raise ValueError("horizon must be positive")
    if float(dt) <= 0.0:
        raise ValueError("dt must be positive")
    if float(q0) == 0.0 and float(p0) == 0.0:
        raise ValueError("initial harmonic energy must be positive")
    return float(q0), float(p0), float(horizon), float(dt)


def _integrate_harmonic(
    solver: RK4Solver | StormerVerlet,
    q0: float,
    p0: float,
    horizon: float,
    *,
    dt: float = 1e-2,
) -> dict[str, np.ndarray[Any, Any] | float]:
    q0, p0, horizon, dt = _validate_harmonic_inputs(q0, p0, horizon, dt)
    steps = max(1, int(round(horizon / dt)))
    y = np.array([q0, p0], dtype=np.float64)
    t = 0.0
    initial_energy = _harmonic_energy(y)

    for _ in range(steps):
        y, dt_used = solver.step(_harmonic_oscillator_rhs, y, t, dt)
        t += dt_used

    phase = float(math.atan2(y[1], y[0]))
    final_energy = _harmonic_energy(y)
    return {
        "state": y,
        "phase": phase,
        "final_energy": final_energy,
        "relative_energy_drift": abs(final_energy - initial_energy) / initial_energy,
    }


def _harmonic_rk4_baseline(
    q0: float,
    p0: float,
    horizon: float,
    *,
    dt: float = 1e-2,
) -> dict[str, np.ndarray[Any, Any] | float]:
    return _integrate_harmonic(RK4Solver(), q0, p0, horizon, dt=dt)


def _harmonic_stormer_verlet_candidate(
    q0: float,
    p0: float,
    horizon: float,
    *,
    dt: float = 1e-2,
) -> dict[str, np.ndarray[Any, Any] | float]:
    return _integrate_harmonic(StormerVerlet(), q0, p0, horizon, dt=dt)


def make_harmonic_symplectic_route() -> AlternativePathRoute[
    dict[str, np.ndarray[Any, Any] | float]
]:
    """Route harmonic-oscillator integration against the symplectic solver."""
    return AlternativePathRoute(
        name="physics.oscillator.harmonic-symplectic",
        baseline=_harmonic_rk4_baseline,
        candidate=_harmonic_stormer_verlet_candidate,
        summary="RK4 baseline vs Störmer-Verlet candidate on a harmonic oscillator",
        expected_behavior=(
            "Candidate should remain close to the RK4 baseline on bounded horizons "
            "while keeping relative energy drift low on this Hamiltonian system"
        ),
    )


def _kuramoto_phase_velocity(
    phases: np.ndarray[Any, Any],
    omegas: np.ndarray[Any, Any],
    coupling: float,
) -> np.ndarray[Any, Any]:
    n = phases.size
    phase_diff = phases[np.newaxis, :] - phases[:, np.newaxis]
    coupling_term = coupling * np.sum(np.sin(phase_diff), axis=1) / n
    derivative: np.ndarray[Any, Any] = omegas + coupling_term
    return derivative


def _kuramoto_order_parameter(phases: np.ndarray[Any, Any]) -> float:
    return float(np.abs(np.mean(np.exp(1j * phases))))


def _kuramoto_interaction_energy(phases: np.ndarray[Any, Any], coupling: float) -> float:
    phase_diff = phases[np.newaxis, :] - phases[:, np.newaxis]
    return float(-0.5 * coupling * np.mean(np.cos(phase_diff)))


def _validate_kuramoto_route_inputs(
    initial_phases: np.ndarray[Any, Any],
    horizon: float,
    omegas: np.ndarray[Any, Any],
    coupling: float,
    dt: float,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], float, float, float, int]:
    phases = np.asarray(initial_phases, dtype=np.float64)
    omega_arr = np.asarray(omegas, dtype=np.float64)

    if phases.ndim != 1 or phases.size == 0:
        raise ValueError("initial_phases must be a non-empty 1-D array")
    if omega_arr.shape != phases.shape:
        raise ValueError("omegas must be a 1-D array matching initial_phases")
    if not np.all(np.isfinite(phases)):
        raise ValueError("initial_phases must contain only finite values")
    if not np.all(np.isfinite(omega_arr)):
        raise ValueError("omegas must contain only finite values")
    if isinstance(horizon, bool) or not isinstance(horizon, int | float):
        raise ValueError("horizon must be finite and positive")
    if not math.isfinite(float(horizon)) or float(horizon) <= 0.0:
        raise ValueError("horizon must be finite and positive")
    if isinstance(coupling, bool) or not isinstance(coupling, int | float):
        raise ValueError("coupling must be finite and non-negative")
    if not math.isfinite(float(coupling)) or float(coupling) < 0.0:
        raise ValueError("coupling must be finite and non-negative")
    if isinstance(dt, bool) or not isinstance(dt, int | float):
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError("dt must be finite and positive")

    steps = max(1, int(round(float(horizon) / float(dt))))
    return phases.copy(), omega_arr.copy(), float(horizon), float(coupling), float(dt), steps


def _wrap_phases(phases: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    wrapped: np.ndarray[Any, Any] = np.mod(phases, 2.0 * np.pi)
    return wrapped


def _kuramoto_euler_baseline(
    initial_phases: np.ndarray[Any, Any],
    horizon: float,
    *,
    omegas: np.ndarray[Any, Any],
    coupling: float,
    dt: float = 1e-3,
) -> dict[str, np.ndarray[Any, Any] | float]:
    phases, omega_arr, _, coupling, dt, steps = _validate_kuramoto_route_inputs(
        initial_phases, horizon, omegas, coupling, dt
    )
    initial_order = _kuramoto_order_parameter(phases)
    initial_energy = _kuramoto_interaction_energy(phases, coupling)

    for _ in range(steps):
        phases = _wrap_phases(phases + dt * _kuramoto_phase_velocity(phases, omega_arr, coupling))

    final_order = _kuramoto_order_parameter(phases)
    final_energy = _kuramoto_interaction_energy(phases, coupling)
    return {
        "phases": phases,
        "order_parameter": final_order,
        "order_parameter_drift": abs(final_order - initial_order),
        "interaction_energy": final_energy,
        "interaction_energy_drift": abs(final_energy - initial_energy),
    }


def _kuramoto_xy_lift_candidate(
    initial_phases: np.ndarray[Any, Any],
    horizon: float,
    *,
    omegas: np.ndarray[Any, Any],
    coupling: float,
    dt: float = 1e-3,
) -> dict[str, np.ndarray[Any, Any] | float]:
    phases, omega_arr, _, coupling, dt, steps = _validate_kuramoto_route_inputs(
        initial_phases, horizon, omegas, coupling, dt
    )
    initial_order = _kuramoto_order_parameter(phases)
    initial_energy = _kuramoto_interaction_energy(phases, coupling)
    initial_momenta = _kuramoto_phase_velocity(phases, omega_arr, coupling)
    solver = StormerVerlet()

    def rhs(_t: float, y: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        n = omega_arr.size
        q = y[:n]
        p = y[n:]
        phase_accel = _kuramoto_phase_velocity(q, omega_arr, coupling)
        return np.concatenate([p, phase_accel])

    t = 0.0
    state = np.concatenate([phases, initial_momenta])
    for _ in range(steps):
        state, dt_used = solver.step(rhs, state, t, dt)
        state[: omega_arr.size] = _wrap_phases(state[: omega_arr.size])
        t += dt_used

    final_phases = state[: omega_arr.size]
    final_order = _kuramoto_order_parameter(final_phases)
    final_energy = _kuramoto_interaction_energy(final_phases, coupling)
    return {
        "phases": final_phases,
        "order_parameter": final_order,
        "order_parameter_drift": abs(final_order - initial_order),
        "interaction_energy": final_energy,
        "interaction_energy_drift": abs(final_energy - initial_energy),
    }


def make_kuramoto_noiseless_symplectic_lift_route() -> AlternativePathRoute[
    dict[str, np.ndarray[Any, Any] | float]
]:
    """Route a bounded noiseless Kuramoto regime against a symplectic XY lift."""
    return AlternativePathRoute(
        name="physics.kuramoto.noiseless-symplectic-lift",
        baseline=_kuramoto_euler_baseline,
        candidate=_kuramoto_xy_lift_candidate,
        summary="Noiseless Kuramoto Euler baseline vs symplectic XY-lift candidate",
        expected_behavior=(
            "On short noiseless horizons the lifted candidate should stay close in "
            "phase and order parameter while remaining explicitly separate from the "
            "production Euler solver"
        ),
    )
