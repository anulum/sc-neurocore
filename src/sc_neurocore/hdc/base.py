# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hyperdimensional Computing Encoder

"""Hyperdimensional computing encoder and associative clean-up memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class HDCEncoder:
    """Hyperdimensional computing encoder.

    Dimension D is usually >= 10,000.
    """

    dim: int = 10000

    def generate_random_vector(self) -> np.ndarray[Any, Any]:
        """Generate a random D-dimensional binary vector in {0, 1}."""
        # We use {0, 1} for compatibility with our SC
        vector: np.ndarray[Any, Any] = np.random.randint(0, 2, self.dim).astype(np.uint8)
        return vector

    def bind(self, v1: np.ndarray[Any, Any], v2: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Bind two hypervectors via XOR."""
        bound: np.ndarray[Any, Any] = np.bitwise_xor(v1, v2)
        return bound

    def bundle(self, vectors: list[np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
        """Bundle hypervectors by majority superposition."""
        if not vectors:
            return np.zeros(self.dim, dtype=np.uint8)

        # Sum columns
        sum_vec = np.sum(vectors, axis=0)
        threshold = len(vectors) / 2.0

        bundled: np.ndarray[Any, Any] = (sum_vec > threshold).astype(np.uint8)
        return bundled

    def permute(self, v: np.ndarray[Any, Any], shifts: int = 1) -> np.ndarray[Any, Any]:
        """Permute a hypervector by a cyclic shift."""
        shifted: np.ndarray[Any, Any] = np.roll(v, shifts)
        return shifted


@dataclass
class AssociativeMemory:
    """Simple HDC associative clean-up memory.

    Stores (key, value) pairs or bare prototypes for nearest-match retrieval.
    """

    memory: dict[str, Any] = field(default_factory=dict)

    def store(self, label: str, vector: np.ndarray[Any, Any]) -> None:
        """Store a labelled hypervector in the clean-up memory."""
        self.memory[label] = vector

    def query(self, query_vec: np.ndarray[Any, Any]) -> str | None:
        """Return the label of the closest stored vector by Hamming distance."""
        best_label = None
        min_dist = float("inf")

        for label, mem_vec in self.memory.items():
            # Hamming distance = count(XOR)
            dist = float(np.count_nonzero(np.bitwise_xor(query_vec, mem_vec)))
            if dist < min_dist:
                min_dist = dist
                best_label = label

        return best_label
