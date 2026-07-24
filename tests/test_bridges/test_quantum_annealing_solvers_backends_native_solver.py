# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (native_solver) from former test_quantum_annealing_solvers_backends.py

from __future__ import annotations

from quantum_annealing_solvers_backends_support import *  # noqa: F403


def test_native_solver_contract_and_seed_forwarding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Native dispatch validates and maps arrays while forwarding the configured seed."""
    captured: tuple[object, ...] = ()

    def fake_solver(*args: object) -> Mapping[str, object]:
        nonlocal captured
        captured = args
        return _valid_native_result(12)

    monkeypatch.setattr(backends, "HAS_RUST_QA", True)
    monkeypatch.setattr(backends, "_rust_simulated_annealing", fake_solver)
    model = IsingModel(
        h={0: 0.5, 2: -0.25},
        J={(0, 1): -1.0, (1, 2): 0.75},
        offset=0.5,
        n_qubits=12,
    )
    result = SimulatedAnnealer(
        n_sweeps=17,
        beta_start=0.2,
        beta_end=3.0,
        seed=91,
        backend="rust",
    ).solve_ising(model, 2)
    assert result["backend"] == "rust"
    assert result["best_spins"][1] == -1
    assert result["samples"][1][0] == -1
    assert captured[-1] == 91
    assert captured[:7] == (
        [0, 2],
        [0.5, -0.25],
        [0, 1],
        [1, 2],
        [-1.0, 0.75],
        12,
        0.5,
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda result: result.update(best_spins="bad"), "best_spins"),
        (lambda result: result.update(best_spins=[1]), "best_spins"),
        (lambda result: result.update(samples="bad"), "samples"),
        (lambda result: result.update(samples=[[0] * 12]), "samples"),
        (lambda result: result.update(energies="bad"), "energies"),
        (lambda result: result.update(energies=[float("nan"), -2.0]), "non-finite"),
        (lambda result: result.update(samples=[[1] * 12], energies=[1.0, 2.0]), "mismatched"),
        (lambda result: result.update(best_energy=True), "best_energy"),
    ],
)
def test_native_solver_rejects_malformed_results(
    monkeypatch: pytest.MonkeyPatch,
    mutation: object,
    match: str,
) -> None:
    """Malformed native payloads cannot cross the Python boundary."""
    result = _valid_native_result(12)
    unsafe(mutation)(result)
    monkeypatch.setattr(backends, "HAS_RUST_QA", True)
    monkeypatch.setattr(backends, "_rust_simulated_annealing", lambda *args: result)
    with pytest.raises(RuntimeError, match=match):
        SimulatedAnnealer(backend="rust").solve_ising(IsingModel(h={0: 0.0}, n_qubits=12), 2)


def test_explicit_missing_native_solver_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit native solver request never changes backend silently."""
    monkeypatch.setattr(backends, "HAS_RUST_QA", False)
    with pytest.raises(RuntimeError, match="unavailable"):
        SimulatedAnnealer(backend="rust").solve_ising(simple_ising())
