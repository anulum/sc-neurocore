# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (python_solver) from former test_quantum_annealing_solvers_backends.py

from __future__ import annotations

from quantum_annealing_solvers_backends_support import *  # noqa: F403


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_sweeps": 0}, "n_sweeps"),
        ({"n_sweeps": unsafe(True)}, "n_sweeps"),
        ({"beta_start": 0.0}, "beta_start"),
        ({"beta_end": float("nan")}, "beta_end"),
        ({"beta_start": 2.0, "beta_end": 1.0}, "beta_end"),
        ({"seed": unsafe(1.5)}, "seed"),
        ({"backend": unsafe("gpu")}, "backend"),
    ],
)
def test_simulated_annealer_rejects_invalid_configuration(
    kwargs: dict[str, object], match: str
) -> None:
    """Annealing schedule and backend configuration fail closed."""
    with pytest.raises(ValueError, match=match):
        SimulatedAnnealer(**unsafe(kwargs))


def test_python_solver_is_deterministic_and_finds_ground_state() -> None:
    """Seeded Python runs preserve the sample contract and find a simple ground state."""
    model = IsingModel(h={0: 0.0, 1: 0.0}, J={(0, 1): -1.0})
    first = SimulatedAnnealer(n_sweeps=200, seed=7, backend="python").solve_ising(
        model, num_reads=5
    )
    second = SimulatedAnnealer(n_sweeps=200, seed=7, backend="python").solve_ising(
        model, num_reads=5
    )
    assert first == second
    assert first["backend"] == "python"
    assert first["best_energy"] == pytest.approx(-1.0)
    assert len(first["samples"]) == 5
    assert all(spin in {-1, 1} for spin in first["best_spins"].values())


def test_one_sweep_python_solver_and_qubo_mapping() -> None:
    """The single-sweep branch and QUBO bit conversion remain valid."""
    qubo = QUBOModel(Q={(0, 0): -1.0, (1, 1): -1.0, (0, 1): 2.0})
    result = SimulatedAnnealer(n_sweeps=1, seed=42, backend="python").solve_qubo(qubo, num_reads=3)
    assert result["backend"] == "python"
    assert len(result["samples"]) == 3
    assert result["best_energy"] == qubo.energy(result["best_bits"])
    assert all(bit in {0, 1} for bit in result["best_bits"].values())


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: SimulatedAnnealer().solve_ising(unsafe("bad")), "IsingModel"),
        (lambda: SimulatedAnnealer().solve_ising(IsingModel()), "one qubit"),
        (lambda: SimulatedAnnealer().solve_ising(simple_ising(), 0), "num_reads"),
        (lambda: SimulatedAnnealer().solve_qubo(unsafe("bad")), "QUBOModel"),
    ],
)
def test_solver_rejects_invalid_calls(call: object, match: str) -> None:
    """Solver entry points validate models and read counts."""
    with pytest.raises(ValueError, match=match):
        unsafe(call)()
