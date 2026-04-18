# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hyperdimensional Computing Encoder

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class HDCEncoder:
    """
    Hyperdimensional Computing Encoder.
    Dimension D usually >= 10,000.
    """

    dim: int = 10000

    def generate_random_vector(self) -> np.ndarray:
        """Generates a random D-dimensional bipolar vector {-1, 1} or {0, 1}."""
        # We use {0, 1} for compatibility with our SC
        return np.random.randint(0, 2, self.dim).astype(np.uint8)

    def bind(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """XOR Binding operation."""
        return np.bitwise_xor(v1, v2)

    def bundle(self, vectors: list[np.ndarray]) -> np.ndarray:
        """
        Majority Bundling (Superposition).
        """
        if not vectors:
            return np.zeros(self.dim, dtype=np.uint8)

        # Sum columns
        sum_vec = np.sum(vectors, axis=0)
        threshold = len(vectors) / 2.0

        return (sum_vec > threshold).astype(np.uint8)

    def permute(self, v: np.ndarray, shifts: int = 1) -> np.ndarray:
        """Cyclic shift (Permutation)."""
        return np.roll(v, shifts)


@dataclass
class AssociativeMemory:
    """
    Simple HDC Associative Memory (Clean-Up Memory).
    Stores (Key, Value) pairs or just prototypes.
    """

    memory: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pass

    def store(self, label: str, vector: np.ndarray) -> None:
        self.memory[label] = vector

    def query(self, query_vec: np.ndarray) -> str | None:
        """Returns label of closest vector (Hamming Distance)."""
        best_label = None
        min_dist = float("inf")

        for label, mem_vec in self.memory.items():
            # Hamming distance = count(XOR)
            dist = np.count_nonzero(np.bitwise_xor(query_vec, mem_vec))
            if dist < min_dist:
                min_dist = dist  # type: ignore[assignment]
                best_label = label

        return best_label
