# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum-annealing model contracts

"""Validated Ising and QUBO value objects."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, cast

from sc_neurocore.bridges import annealing_backends as backends


BackendChoice = Literal["auto", "python", "rust"]


def validate_backend_choice(backend: str) -> BackendChoice:
    """Validate and narrow an execution-backend selector."""
    if backend not in {"auto", "python", "rust"}:
        raise ValueError("backend must be 'auto', 'python', or 'rust'")
    return cast(BackendChoice, backend)


def _require_finite(name: str, value: float) -> float:
    """Return a finite float or raise a field-specific error."""
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _require_index(name: str, value: object) -> int:
    """Return a non-negative integer index."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


class ProblemType(Enum):
    """Quantum optimization problem type."""

    ISING = "ising"
    QUBO = "qubo"


@dataclass
class QubitSpec:
    """Specification for one logical qubit."""

    index: int
    label: str
    bias: float = 0.0

    def __post_init__(self) -> None:
        """Validate the qubit index, label, and bias."""
        self.index = _require_index("index", self.index)
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        self.bias = _require_finite("bias", self.bias)


@dataclass
class CouplerSpec:
    """Specification for one logical Ising/QUBO coupling."""

    qubit_a: int
    qubit_b: int
    strength: float = 0.0

    def __post_init__(self) -> None:
        """Validate distinct endpoints and a finite strength."""
        self.qubit_a = _require_index("qubit_a", self.qubit_a)
        self.qubit_b = _require_index("qubit_b", self.qubit_b)
        if self.qubit_a == self.qubit_b:
            raise ValueError("coupler endpoints must be distinct")
        if self.qubit_a > self.qubit_b:
            self.qubit_a, self.qubit_b = self.qubit_b, self.qubit_a
        self.strength = _require_finite("strength", self.strength)


@dataclass
class IsingModel:
    """Ising spin-glass model ``H = Σhᵢsᵢ + ΣJᵢⱼsᵢsⱼ``."""

    h: dict[int, float] = field(default_factory=dict)
    J: dict[tuple[int, int], float] = field(default_factory=dict)
    offset: float = 0.0
    qubit_labels: dict[int, str] = field(default_factory=dict)
    n_qubits: int = 0
    source: str = ""

    def __post_init__(self) -> None:
        """Normalize canonical couplings and validate model bounds."""
        normalized_h: dict[int, float] = {}
        highest_index = -1
        for raw_index, raw_bias in self.h.items():
            index = _require_index("h index", raw_index)
            normalized_h[index] = _require_finite(f"h[{index}]", raw_bias)
            highest_index = max(highest_index, index)

        normalized_j: dict[tuple[int, int], float] = {}
        for raw_pair, raw_strength in self.J.items():
            if not isinstance(raw_pair, tuple) or len(raw_pair) != 2:
                raise ValueError("J keys must be two-index tuples")
            first = _require_index("J endpoint", raw_pair[0])
            second = _require_index("J endpoint", raw_pair[1])
            if first == second:
                raise ValueError("Ising couplings must connect distinct qubits")
            pair = (min(first, second), max(first, second))
            strength = _require_finite(f"J[{pair}]", raw_strength)
            normalized_j[pair] = normalized_j.get(pair, 0.0) + strength
            highest_index = max(highest_index, *pair)

        normalized_labels: dict[int, str] = {}
        for raw_index, label in self.qubit_labels.items():
            index = _require_index("qubit label index", raw_index)
            if not isinstance(label, str) or not label.strip():
                raise ValueError("qubit labels must be non-empty strings")
            normalized_labels[index] = label
            highest_index = max(highest_index, index)
        if len(set(normalized_labels.values())) != len(normalized_labels):
            raise ValueError("qubit labels must be unique")

        if isinstance(self.n_qubits, bool) or not isinstance(self.n_qubits, int):
            raise ValueError("n_qubits must be a non-negative integer")
        if self.n_qubits < 0:
            raise ValueError("n_qubits must be a non-negative integer")
        if self.n_qubits == 0 and highest_index >= 0:
            self.n_qubits = highest_index + 1
        if highest_index >= self.n_qubits:
            raise ValueError("model indices must be smaller than n_qubits")
        if not isinstance(self.source, str):
            raise ValueError("source must be a string")

        self.h = normalized_h
        self.J = {pair: value for pair, value in normalized_j.items() if value != 0.0}
        self.qubit_labels = normalized_labels
        self.offset = _require_finite("offset", self.offset)

    def energy(
        self,
        spins: Mapping[int, int],
        *,
        backend: BackendChoice = "auto",
    ) -> float:
        """Compute energy for a partial spin assignment.

        Missing spins retain the historical ``+1`` default. An explicit Rust
        request fails when the native engine is unavailable; ``auto`` uses it
        only for models larger than 20 qubits.
        """
        selected = validate_backend_choice(backend)
        for index, spin in spins.items():
            _require_index("spin index", index)
            if spin not in {-1, 1}:
                raise ValueError("spin values must be -1 or +1")

        use_rust = selected == "rust" or (
            selected == "auto" and backends.HAS_RUST_QA and self.n_qubits > 20
        )
        if use_rust:
            kernel = backends.require_rust_energy()
            h_indices = list(self.h)
            j_pairs = list(self.J)
            return float(
                kernel(
                    h_indices,
                    [self.h[index] for index in h_indices],
                    [pair[0] for pair in j_pairs],
                    [pair[1] for pair in j_pairs],
                    [self.J[pair] for pair in j_pairs],
                    [spins.get(index, 1) for index in range(self.n_qubits)],
                    self.offset,
                )
            )

        energy = self.offset
        for index, bias in self.h.items():
            energy += bias * spins.get(index, 1)
        for (first, second), strength in self.J.items():
            energy += strength * spins.get(first, 1) * spins.get(second, 1)
        return energy


@dataclass
class QUBOModel:
    """Quadratic unconstrained binary optimization model ``xᵀQx``."""

    Q: dict[tuple[int, int], float] = field(default_factory=dict)
    offset: float = 0.0
    qubit_labels: dict[int, str] = field(default_factory=dict)
    n_qubits: int = 0
    source: str = ""

    def __post_init__(self) -> None:
        """Normalize matrix keys and validate model bounds."""
        normalized_q: dict[tuple[int, int], float] = {}
        highest_index = -1
        for raw_pair, raw_value in self.Q.items():
            if not isinstance(raw_pair, tuple) or len(raw_pair) != 2:
                raise ValueError("Q keys must be two-index tuples")
            first = _require_index("Q index", raw_pair[0])
            second = _require_index("Q index", raw_pair[1])
            pair = (min(first, second), max(first, second))
            value = _require_finite(f"Q[{pair}]", raw_value)
            normalized_q[pair] = normalized_q.get(pair, 0.0) + value
            highest_index = max(highest_index, *pair)

        normalized_labels: dict[int, str] = {}
        for raw_index, label in self.qubit_labels.items():
            index = _require_index("qubit label index", raw_index)
            if not isinstance(label, str) or not label.strip():
                raise ValueError("qubit labels must be non-empty strings")
            normalized_labels[index] = label
            highest_index = max(highest_index, index)
        if len(set(normalized_labels.values())) != len(normalized_labels):
            raise ValueError("qubit labels must be unique")

        if isinstance(self.n_qubits, bool) or not isinstance(self.n_qubits, int):
            raise ValueError("n_qubits must be a non-negative integer")
        if self.n_qubits < 0:
            raise ValueError("n_qubits must be a non-negative integer")
        if self.n_qubits == 0 and highest_index >= 0:
            self.n_qubits = highest_index + 1
        if highest_index >= self.n_qubits:
            raise ValueError("model indices must be smaller than n_qubits")
        if not isinstance(self.source, str):
            raise ValueError("source must be a string")

        self.Q = {pair: value for pair, value in normalized_q.items() if value != 0.0}
        self.qubit_labels = normalized_labels
        self.offset = _require_finite("offset", self.offset)

    def energy(self, bits: Mapping[int, int]) -> float:
        """Compute QUBO energy for a partial binary assignment."""
        for index, bit in bits.items():
            _require_index("bit index", index)
            if bit not in {0, 1}:
                raise ValueError("bit values must be 0 or 1")
        energy = self.offset
        for (first, second), coefficient in self.Q.items():
            energy += coefficient * bits.get(first, 0) * bits.get(second, 0)
        return energy

    def to_ising(self) -> IsingModel:
        """Convert QUBO to an exactly energy-equivalent Ising model."""
        h: dict[int, float] = {}
        couplings: dict[tuple[int, int], float] = {}
        offset = self.offset

        for (first, second), coefficient in self.Q.items():
            if first == second:
                h[first] = h.get(first, 0.0) + coefficient / 2.0
                offset += coefficient / 2.0
                continue
            couplings[(first, second)] = couplings.get((first, second), 0.0) + coefficient / 4.0
            h[first] = h.get(first, 0.0) + coefficient / 4.0
            h[second] = h.get(second, 0.0) + coefficient / 4.0
            offset += coefficient / 4.0

        return IsingModel(
            h=h,
            J=couplings,
            offset=offset,
            qubit_labels=dict(self.qubit_labels),
            n_qubits=self.n_qubits,
            source=f"{self.source} (QUBO→Ising)",
        )
