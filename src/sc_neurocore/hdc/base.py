# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
from typing import Any, Optional

import numpy as np
from dataclasses import dataclass


@dataclass
class HDCEncoder:
    """
    Hyperdimensional Computing Encoder.
    Dimension D usually >= 10,000.
    """

    dim: int = 10000

    def generate_random_vector(self) -> np.ndarray[Any, Any]:
        """Generates a random D-dimensional bipolar vector {-1, 1} or {0, 1}."""
        # We use {0, 1} for compatibility with our SC
        return np.random.randint(0, 2, self.dim).astype(np.uint8)

    def bind(self, v1: np.ndarray[Any, Any], v2: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """XOR Binding operation."""
        return np.bitwise_xor(v1, v2)

    def bundle(self, vectors: list[np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
        """
        Majority Bundling (Superposition).
        """
        if not vectors:
            return np.zeros(self.dim, dtype=np.uint8)

        # Sum columns
        sum_vec = np.sum(vectors, axis=0)
        threshold = len(vectors) / 2.0

        return (sum_vec > threshold).astype(np.uint8)  # type: ignore

    def permute(self, v: np.ndarray[Any, Any], shifts: int = 1) -> np.ndarray[Any, Any]:
        """Cyclic shift (Permutation)."""
        return np.roll(v, shifts)


@dataclass
class AssociativeMemory:
    """
    Simple HDC Associative Memory (Clean-Up Memory).
    Stores (Key, Value) pairs or just prototypes.
    """

    memory: dict[str, Any] = None  # type: ignore

    def __post_init__(self):  # type: ignore
        self.memory = {}

    def store(self, label: str, vector: np.ndarray[Any, Any]):  # type: ignore
        self.memory[label] = vector

    def query(self, query_vec: np.ndarray[Any, Any]) -> str:
        """Returns label of closest vector (Hamming Distance)."""
        best_label = None
        min_dist = float("inf")

        for label, mem_vec in self.memory.items():
            # Hamming distance = count(XOR)
            dist = np.count_nonzero(np.bitwise_xor(query_vec, mem_vec))
            if dist < min_dist:
                min_dist = dist
                best_label = label

        return best_label  # type: ignore
