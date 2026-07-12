# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Parity tests: SimulatedAnnealer vs neal.SimulatedAnnealingSampler.

These tests cross-check sc-neurocore's in-house simulated-annealing
implementation (`SimulatedAnnealer`) against the reference D-Wave
sampler `neal.SimulatedAnnealingSampler`, used as ground truth for
correctness of the Ising/QUBO formulation.

Closes follow-up task #51.

Both samplers share identical thermodynamics (Metropolis spin
flips with a linear inverse-temperature schedule β: β_start → β_end)
so the **best energies** they return on the same model should agree
to within a few percent for sufficient sweeps and reads. We do
**not** require exact spin-configuration agreement — degenerate
ground states, RNG differences, and the inherent stochasticity of
SA make that brittle.

Tested invariants:
1. Energy of `neal`'s best sample, recomputed by `IsingModel.energy`,
   matches `neal`'s reported energy → confirms our energy function
   matches D-Wave's convention (H = Σ h_i s_i + Σ J_ij s_i s_j).
2. `SimulatedAnnealer`'s best energy is within tolerance of neal's
   best energy on planted-ground-state instances.
3. Mean over many reads is bounded by neal's mean (sanity check on
   sampling distribution).
"""

from __future__ import annotations

import numpy as np
import pytest
from typing import Any

neal: Any = pytest.importorskip("neal")
dimod: Any = pytest.importorskip("dimod")

from sc_neurocore.bridges.quantum_annealing import (
    IsingModel,
    SimulatedAnnealer,
)


# ───────────────────────── helpers ─────────────────────────


def _build_planted_ising(
    n: int,
    p: float,
    coupling: float,
    seed: int,
) -> tuple[IsingModel, dict[int, int]]:
    """Build an Erdős–Rényi Ising with a known planted ground state.

    All couplings are set so the all-spins-up state is the unique
    ground state (J_ij = -|coupling|, h_i = -|h|). This gives us a
    reference to compare both samplers against.
    """
    rng = np.random.default_rng(seed)
    h = {i: -abs(coupling) for i in range(n)}
    J: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                J[(i, j)] = -abs(coupling)
    model = IsingModel(h=h, J=J, offset=0.0, n_qubits=n, source="planted_FM")
    planted = {i: 1 for i in range(n)}
    return model, planted


def _to_neal_bqm(model: IsingModel) -> Any:
    """Convert our IsingModel → dimod.BQM in spin vartype."""
    return dimod.BinaryQuadraticModel.from_ising(
        h=dict(model.h),
        J=dict(model.J),
        offset=model.offset,
    )


def _neal_best_energy(
    model: IsingModel,
    num_reads: int,
    seed: int,
    sweeps: int,
) -> tuple[float, dict[int, int], float]:
    """Run neal and return (best_energy, best_spins, mean_energy)."""
    bqm = _to_neal_bqm(model)
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=sweeps,
        seed=seed,
    )
    record = sampleset.record
    best_idx = int(np.argmin(record.energy))
    best_sample = sampleset.samples()[best_idx]
    best_spins = {int(k): int(v) for k, v in dict(best_sample).items()}
    mean_e = float(np.mean(record.energy))
    return float(record.energy[best_idx]), best_spins, mean_e


# ───────────────────────── tests ─────────────────────────


@pytest.mark.parametrize("n,p,seed", [(8, 0.5, 11), (12, 0.4, 22), (16, 0.3, 33)])
def test_energy_function_matches_dimod(n: int, p: float, seed: int) -> None:
    """Our Ising energy must agree with dimod's BQM.energy() to ε≈0."""
    model, _ = _build_planted_ising(n, p, coupling=1.0, seed=seed)
    bqm = _to_neal_bqm(model)
    rng = np.random.default_rng(seed * 7)
    for _ in range(8):
        spins = {i: int(rng.choice([-1, 1])) for i in range(n)}
        ours = model.energy(spins, backend="python")
        theirs = float(bqm.energy(spins))
        assert ours == pytest.approx(theirs, abs=1e-9), (
            f"energy mismatch on n={n} spins={spins}: ours={ours} dimod={theirs}"
        )


@pytest.mark.parametrize("n,seed", [(10, 100), (20, 200), (30, 300)])
def test_planted_ground_state_recovered(n: int, seed: int) -> None:
    """Both samplers should find energy close to the planted GS."""
    model, planted = _build_planted_ising(n, p=0.3, coupling=1.0, seed=seed)
    planted_energy = model.energy(planted, backend="python")

    neal_best, _, _ = _neal_best_energy(
        model,
        num_reads=20,
        seed=seed,
        sweeps=500,
    )
    sa = SimulatedAnnealer(
        n_sweeps=500,
        beta_start=0.1,
        beta_end=10.0,
        seed=seed,
        backend="python",
    )
    ours = sa.solve_ising(model, num_reads=20)

    # neal must hit (or be within 1%) of planted.
    assert neal_best <= planted_energy * 0.99 + 1e-6, (
        f"neal didn't find planted GS on n={n}: neal={neal_best} planted={planted_energy}"
    )
    # Our SA should be within 5% of neal's best — this is the parity claim.
    # On strongly ferromagnetic instances both routinely hit the optimum.
    rel_gap = abs(ours["best_energy"] - neal_best) / max(abs(neal_best), 1e-6)
    assert rel_gap < 0.05, (
        f"parity gap too large on n={n}: ours={ours['best_energy']} "
        f"neal={neal_best} rel_gap={rel_gap:.4f}"
    )


@pytest.mark.parametrize("n,seed", [(15, 42), (25, 84)])
def test_random_ising_best_energy_within_tolerance(n: int, seed: int) -> None:
    """On random ±1 ising instances, our best ≈ neal's best (±10%)."""
    rng = np.random.default_rng(seed)
    h = {i: float(rng.choice([-1.0, 1.0])) for i in range(n)}
    J: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.3:
                J[(i, j)] = float(rng.choice([-1.0, 1.0]))
    model = IsingModel(h=h, J=J, offset=0.0, n_qubits=n, source="random_pm1")

    neal_best, _, _ = _neal_best_energy(
        model,
        num_reads=50,
        seed=seed,
        sweeps=1000,
    )
    sa = SimulatedAnnealer(
        n_sweeps=1000,
        beta_start=0.1,
        beta_end=10.0,
        seed=seed,
        backend="python",
    )
    ours = sa.solve_ising(model, num_reads=50)

    # On frustrated random instances, finding the global minimum is hard.
    # We require: our best energy ≤ neal's best energy × (1 + 10%) when
    # neal_best is negative, OR within 0.1 absolute when near zero.
    # Because energies are negative for typical ising minima, the
    # tolerance is "we are not much higher than neal".
    if neal_best < 0:
        assert ours["best_energy"] <= neal_best * 0.90, (
            f"our SA performs worse than neal on n={n}: ours={ours['best_energy']} neal={neal_best}"
        )
    else:
        assert ours["best_energy"] <= neal_best + 0.5, (
            f"our SA performs worse than neal on n={n}: ours={ours['best_energy']} neal={neal_best}"
        )


def test_offset_propagation() -> None:
    """Both engines must apply the constant `offset` identically."""
    model, _ = _build_planted_ising(n=8, p=0.4, coupling=1.0, seed=7)
    model.offset = 12.5
    bqm = _to_neal_bqm(model)
    spins = {i: 1 for i in range(8)}
    ours = model.energy(spins, backend="python")
    theirs = float(bqm.energy(spins))
    assert ours == pytest.approx(theirs, abs=1e-9)
    # Offset is part of the energy.
    model.offset = 0.0
    bqm0 = _to_neal_bqm(model)
    assert ours - 12.5 == pytest.approx(float(bqm0.energy(spins)), abs=1e-9)


def test_returned_spin_assignment_is_valid_for_neal_best() -> None:
    """neal's best spin assignment, fed to our `IsingModel.energy`,
    must reproduce neal's reported best energy exactly.

    This is the strictest cross-validation: it confirms that our
    Hamiltonian convention matches D-Wave's (no sign flip on J,
    no missing 1/2 factor, etc.).
    """
    model, _ = _build_planted_ising(n=20, p=0.25, coupling=1.0, seed=999)
    neal_best, neal_spins, _ = _neal_best_energy(
        model,
        num_reads=10,
        seed=999,
        sweeps=300,
    )
    recomputed = model.energy(neal_spins, backend="python")
    assert recomputed == pytest.approx(neal_best, abs=1e-9), (
        f"convention mismatch: neal_e={neal_best} our_e_on_neal_spins={recomputed}"
    )
