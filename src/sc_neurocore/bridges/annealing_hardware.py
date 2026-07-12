# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Annealer hardware and embedding utilities

"""Hardware-capacity estimates and broken-chain post-processing."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from sc_neurocore.bridges.annealing_models import IsingModel


def _validated_chains(
    chains: Mapping[int, Sequence[int]],
) -> dict[int, tuple[int, ...]]:
    """Validate a one-to-one logical-to-physical chain mapping."""
    normalized: dict[int, tuple[int, ...]] = {}
    claimed_physical: set[int] = set()
    for logical, physical_sequence in chains.items():
        if isinstance(logical, bool) or not isinstance(logical, int) or logical < 0:
            raise ValueError("logical chain indices must be non-negative integers")
        if isinstance(physical_sequence, (str, bytes)) or not physical_sequence:
            raise ValueError("every logical qubit must map to a non-empty physical chain")
        physical = tuple(physical_sequence)
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in physical
        ):
            raise ValueError("physical chain indices must be non-negative integers")
        if len(set(physical)) != len(physical):
            raise ValueError("a physical chain must not contain duplicate qubits")
        if claimed_physical.intersection(physical):
            raise ValueError("physical qubits must not belong to multiple logical chains")
        normalized[logical] = physical
        claimed_physical.update(physical)
    return normalized


def _validated_physical_samples(
    samples: Sequence[Mapping[int, int]],
) -> list[dict[int, int]]:
    """Validate physical samples without requiring every chain qubit."""
    if isinstance(samples, (str, bytes)):
        raise ValueError("physical_samples must be a sequence of mappings")
    normalized: list[dict[int, int]] = []
    for sample in samples:
        row: dict[int, int] = {}
        for index, spin in sample.items():
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise ValueError("physical sample indices must be non-negative integers")
            if spin not in {-1, 1}:
                raise ValueError("physical sample spins must be -1 or +1")
            row[index] = spin
        normalized.append(row)
    return normalized


class HardwareGraph:
    """Capacity model for Chimera, Pegasus, and Zephyr topologies."""

    _TOPOLOGIES: ClassVar[dict[str, dict[str, int]]] = {
        "chimera": {"connectivity": 6, "base_qubits_per_cell": 8},
        "pegasus": {"connectivity": 15, "base_qubits_per_cell": 24},
        "zephyr": {"connectivity": 20, "base_qubits_per_cell": 48},
    }

    def __init__(self, topology: str = "pegasus", size: int = 16) -> None:
        """Select a topology and a valid positive size parameter."""
        if topology not in self._TOPOLOGIES:
            raise ValueError(f"Unknown topology: {topology}")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError("size must be a positive integer")
        if topology == "pegasus" and size < 2:
            raise ValueError("Pegasus size must be at least two")
        self._topology = topology
        self._size = size
        self._props = self._TOPOLOGIES[topology]

    @property
    def n_physical_qubits(self) -> int:
        """Return the idealized physical-qubit capacity."""
        if self._topology == "chimera":
            return self._size * self._size * 8
        if self._topology == "pegasus":
            return 24 * self._size * (self._size - 1)
        return 48 * self._size * self._size

    @property
    def connectivity(self) -> int:
        """Return the idealized per-qubit connectivity."""
        return self._props["connectivity"]

    def can_embed(self, model: IsingModel) -> dict[str, Any]:
        """Return a conservative degree-based capacity estimate."""
        if not isinstance(model, IsingModel) or model.n_qubits <= 0:
            raise ValueError("model must be a non-empty IsingModel")
        degree = {index: 0 for index in range(model.n_qubits)}
        for first, second in model.J:
            degree[first] += 1
            degree[second] += 1
        max_degree = max(degree.values())
        chain_length = max(1, math.ceil(max_degree / self.connectivity))
        physical_needed = model.n_qubits * chain_length
        return {
            "embeddable": physical_needed <= self.n_physical_qubits,
            "topology": self._topology,
            "size": self._size,
            "n_logical": model.n_qubits,
            "n_couplers": len(model.J),
            "max_degree": max_degree,
            "chain_length_estimate": chain_length,
            "n_physical_available": self.n_physical_qubits,
            "estimated_physical_needed": physical_needed,
            "utilization_pct": physical_needed / self.n_physical_qubits * 100.0,
        }


class ChainBreakResolver:
    """Resolve embedded-chain disagreement by vote or local energy search."""

    def __init__(self, method: str = "majority_vote") -> None:
        """Select a supported deterministic resolution method."""
        if method not in {"majority_vote", "minimize_energy"}:
            raise ValueError(f"Unknown method: {method}")
        self._method = method

    def resolve(
        self,
        physical_samples: Sequence[Mapping[int, int]],
        chains: Mapping[int, Sequence[int]],
        model: IsingModel | None = None,
    ) -> list[dict[int, int]]:
        """Map physical samples to logical samples and optionally refine them."""
        normalized_samples = _validated_physical_samples(physical_samples)
        normalized_chains = _validated_chains(chains)
        if self._method == "minimize_energy" and model is None:
            raise ValueError("model is required for minimize_energy resolution")
        if model is not None:
            if not isinstance(model, IsingModel):
                raise ValueError("model must be an IsingModel")
            if any(logical >= model.n_qubits for logical in normalized_chains):
                raise ValueError("chain logical indices must fit within model.n_qubits")

        resolved: list[dict[int, int]] = []
        for sample in normalized_samples:
            logical = {
                logical_index: (
                    1
                    if sum(sample.get(physical_index, 1) for physical_index in physical) >= 0
                    else -1
                )
                for logical_index, physical in normalized_chains.items()
            }
            if self._method == "minimize_energy" and model is not None:
                energy = model.energy(logical, backend="python")
                for logical_index in logical:
                    candidate = dict(logical)
                    candidate[logical_index] *= -1
                    candidate_energy = model.energy(candidate, backend="python")
                    if candidate_energy < energy:
                        logical[logical_index] *= -1
                        energy = candidate_energy
            resolved.append(logical)
        return resolved

    def analyze_breaks(
        self,
        physical_samples: Sequence[Mapping[int, int]],
        chains: Mapping[int, Sequence[int]],
    ) -> dict[str, Any]:
        """Measure per-chain and aggregate break rates."""
        normalized_samples = _validated_physical_samples(physical_samples)
        normalized_chains = _validated_chains(chains)
        total_breaks = 0
        breakable_chain_count = 0
        per_chain: dict[int, float] = {}
        for logical_index, physical in normalized_chains.items():
            if len(physical) == 1:
                per_chain[logical_index] = 0.0
                continue
            breaks = sum(
                1
                for sample in normalized_samples
                if len({sample.get(index, 1) for index in physical}) > 1
            )
            per_chain[logical_index] = (
                breaks / len(normalized_samples) if normalized_samples else 0.0
            )
            total_breaks += breaks
            breakable_chain_count += 1
        opportunity_count = breakable_chain_count * len(normalized_samples)
        return {
            "total_breaks": total_breaks,
            "break_rate": total_breaks / opportunity_count if opportunity_count else 0.0,
            "per_chain": per_chain,
            "n_chains": len(normalized_chains),
        }
