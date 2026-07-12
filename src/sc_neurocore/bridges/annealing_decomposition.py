# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Large Ising problem decomposition

"""Deterministic overlapping decomposition and result reconstruction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sc_neurocore.bridges.annealing_models import IsingModel
from sc_neurocore.bridges.annealing_solvers import SimulatedAnnealer


def _positive_int(name: str, value: int) -> int:
    """Return a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class ProblemDecomposer:
    """Partition large Ising graphs into bounded overlapping subproblems."""

    def __init__(
        self,
        max_subproblem_size: int = 64,
        overlap: int = 4,
        n_iterations: int = 10,
    ) -> None:
        """Configure maximum size, real overlap, and merge iterations."""
        self._max_size = _positive_int("max_subproblem_size", max_subproblem_size)
        if isinstance(overlap, bool) or not isinstance(overlap, int) or overlap < 0:
            raise ValueError("overlap must be a non-negative integer")
        if overlap >= self._max_size:
            raise ValueError("overlap must be smaller than max_subproblem_size")
        self._overlap = overlap
        self._n_iterations = _positive_int("n_iterations", n_iterations)

    def _partition_indices(self, model: IsingModel) -> list[list[int]]:
        """Return deterministic graph-aware partitions in global indices."""
        if model.n_qubits <= self._max_size:
            return [list(range(model.n_qubits))]

        neighbors: dict[int, dict[int, float]] = {index: {} for index in range(model.n_qubits)}
        for (first, second), strength in model.J.items():
            magnitude = abs(strength)
            neighbors[first][second] = magnitude
            neighbors[second][first] = magnitude

        remaining = set(range(model.n_qubits))
        assigned: set[int] = set()
        partitions: list[list[int]] = []
        while remaining:
            seed = min(remaining)
            overlap_candidates = sorted(
                (
                    (neighbors[seed].get(index, 0.0), index)
                    for index in assigned
                    if index in neighbors[seed]
                ),
                key=lambda item: (-item[0], item[1]),
            )
            shared = [index for _, index in overlap_candidates[: self._overlap]]
            partition = [*shared, seed]
            remaining.remove(seed)
            assigned.add(seed)

            while len(partition) < self._max_size and remaining:
                scored: list[tuple[float, int]] = []
                for candidate in remaining:
                    score = max(
                        (neighbors[candidate].get(member, 0.0) for member in partition),
                        default=0.0,
                    )
                    if score > 0.0:
                        scored.append((score, candidate))
                next_qubit = (
                    min(remaining)
                    if not scored
                    else min(scored, key=lambda item: (-item[0], item[1]))[1]
                )
                partition.append(next_qubit)
                remaining.remove(next_qubit)
                assigned.add(next_qubit)
            partitions.append(partition)
        return partitions

    @staticmethod
    def _submodel(model: IsingModel, indices: list[int], part_index: int) -> IsingModel:
        """Build one local-indexed model from global indices."""
        local_index = {global_index: index for index, global_index in enumerate(indices)}
        index_set = set(indices)
        couplings: dict[tuple[int, int], float] = {}
        for (first, second), strength in model.J.items():
            if first in index_set and second in index_set:
                local_first = local_index[first]
                local_second = local_index[second]
                couplings[(min(local_first, local_second), max(local_first, local_second))] = (
                    strength
                )
        return IsingModel(
            h={
                local_index[global_index]: model.h.get(global_index, 0.0)
                for global_index in indices
            },
            J=couplings,
            qubit_labels={
                local_index[global_index]: model.qubit_labels.get(global_index, f"q{global_index}")
                for global_index in indices
            },
            n_qubits=len(indices),
            source=f"{model.source}_part{part_index}",
        )

    def decompose(self, model: IsingModel) -> list[IsingModel]:
        """Return bounded submodels; small inputs retain object identity."""
        if not isinstance(model, IsingModel) or model.n_qubits <= 0:
            raise ValueError("model must be a non-empty IsingModel")
        if model.n_qubits <= self._max_size:
            return [model]
        return [
            self._submodel(model, indices, part_index)
            for part_index, indices in enumerate(self._partition_indices(model))
        ]

    def solve_decomposed(
        self,
        model: IsingModel,
        solver: SimulatedAnnealer | None = None,
    ) -> dict[str, Any]:
        """Solve submodels iteratively and reconstruct by exact global index."""
        if not isinstance(model, IsingModel) or model.n_qubits <= 0:
            raise ValueError("model must be a non-empty IsingModel")
        active_solver = solver or SimulatedAnnealer(n_sweeps=1000, seed=42)
        if not isinstance(active_solver, SimulatedAnnealer):
            raise ValueError("solver must be a SimulatedAnnealer")

        partitions = self._partition_indices(model)
        submodels = [
            model
            if len(partitions) == 1 and partitions[0] == list(range(model.n_qubits))
            else self._submodel(model, indices, part_index)
            for part_index, indices in enumerate(partitions)
        ]
        global_spins = {index: 1 for index in range(model.n_qubits)}
        for _ in range(self._n_iterations):
            for indices, submodel in zip(partitions, submodels):
                result = active_solver.solve_ising(submodel, num_reads=5)
                best_spins = result.get("best_spins")
                if not isinstance(best_spins, Mapping):
                    raise RuntimeError("subproblem solver returned no best_spins mapping")
                for local_index, spin in best_spins.items():
                    if (
                        isinstance(local_index, bool)
                        or not isinstance(local_index, int)
                        or not 0 <= local_index < len(indices)
                        or spin not in {-1, 1}
                    ):
                        raise RuntimeError("subproblem solver returned an invalid spin mapping")
                    global_spins[indices[local_index]] = int(spin)

        return {
            "best_spins": global_spins,
            "best_energy": model.energy(global_spins, backend="python"),
            "n_partitions": len(partitions),
            "n_iterations": self._n_iterations,
        }
