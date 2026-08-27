# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hyperdimensional Computing Encoder

"""Hyperdimensional computing encoder and associative clean-up memory.

Binary {0, 1} hypervector algebra (bind = XOR, bundle = majority,
permute = cyclic shift) with a seeded generator, a named item memory,
an explicit bundle tie policy, and linear level (thermometer) encoding
for scalars. All randomness flows through one ``numpy`` generator owned
by the encoder, so a seeded encoder is fully deterministic given the
same call sequence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

_TIE_POLICIES = ("zeros", "ones", "random")


@dataclass
class HDCEncoder:
    """Hyperdimensional computing encoder.

    Dimension D is usually >= 10,000. ``seed`` makes every draw
    deterministic (given the same call order); ``tie_policy`` states
    what an even-count bundle does on exactly tied bit positions:
    ``"zeros"`` clears them (historical strict-majority behaviour),
    ``"ones"`` sets them, and ``"random"`` decides each tied position
    from a fresh seeded tie-break hypervector (the unbiased Kanerva
    convention).
    """

    dim: int = 10000
    seed: int | None = None
    tie_policy: str = "zeros"
    _rng: np.random.Generator = field(init=False, repr=False)
    _item_memory: dict[str, np.ndarray[Any, Any]] = field(
        init=False, repr=False, default_factory=dict
    )
    _level_memory: dict[tuple[float, float, int], np.ndarray[Any, Any]] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        """Validate configuration and initialise the seeded generator."""
        if not isinstance(self.dim, int) or isinstance(self.dim, bool) or self.dim < 1:
            raise ValueError("dim must be a positive integer")
        if self.tie_policy not in _TIE_POLICIES:
            raise ValueError(f"tie_policy must be one of {list(_TIE_POLICIES)}")
        self._rng = np.random.default_rng(self.seed)

    def generate_random_vector(self) -> np.ndarray[Any, Any]:
        """Generate a random D-dimensional binary vector in {0, 1}."""
        # We use {0, 1} for compatibility with our SC
        vector: np.ndarray[Any, Any] = self._rng.integers(0, 2, self.dim, dtype=np.uint8)
        return vector

    def item(self, name: str) -> np.ndarray[Any, Any]:
        """Return the named item hypervector, drawing it on first use.

        The same encoder always returns the identical vector for the
        same name; a seeded encoder reproduces the whole item memory
        when the names are first requested in the same order.
        """
        if name not in self._item_memory:
            self._item_memory[name] = self.generate_random_vector()
        return self._item_memory[name].copy()

    def bind(self, v1: np.ndarray[Any, Any], v2: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Bind two hypervectors via XOR."""
        bound: np.ndarray[Any, Any] = np.bitwise_xor(v1, v2)
        return bound

    def bundle(self, vectors: list[np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
        """Bundle hypervectors by majority superposition.

        Bit positions with a strict majority of ones become one and a
        strict majority of zeros become zero; exactly tied positions
        (possible only for an even vector count) follow ``tie_policy``.
        """
        if not vectors:
            return np.zeros(self.dim, dtype=np.uint8)
        sum_vec = np.sum(vectors, axis=0)
        return self.majority(sum_vec, len(vectors))

    def majority(self, sum_vec: np.ndarray[Any, Any], count: int) -> np.ndarray[Any, Any]:
        """Return the majority vector of ``count`` bundled binary vectors.

        ``sum_vec`` holds the per-position count of ones. This is the
        bundle kernel, shared with the centroid classifier so both
        apply the identical tie policy.
        """
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("count must be a positive integer")
        doubled = 2 * np.asarray(sum_vec, dtype=np.int64)
        majority = (doubled > count).astype(np.uint8)
        if count % 2 == 0:
            tied = doubled == count
            if bool(np.any(tied)):
                if self.tie_policy == "ones":
                    majority[tied] = 1
                elif self.tie_policy == "random":
                    tie_break = self.generate_random_vector()
                    majority[tied] = tie_break[tied]
        bundled: np.ndarray[Any, Any] = majority
        return bundled

    def permute(self, v: np.ndarray[Any, Any], shifts: int = 1) -> np.ndarray[Any, Any]:
        """Permute a hypervector by a cyclic shift."""
        shifted: np.ndarray[Any, Any] = np.roll(v, shifts)
        return shifted

    def level_vectors(self, low: float, high: float, levels: int) -> np.ndarray[Any, Any]:
        """Return the ``levels`` linear level hypervectors for [low, high].

        Level 0 is a fresh random hypervector; each subsequent level
        flips the next ``(dim // 2) // (levels - 1)`` positions of a
        fixed seeded permutation, so the Hamming distance between two
        levels grows linearly with their separation and the endpoints
        differ in ``(dim // 2) // (levels - 1) * (levels - 1)`` bits
        (approaching orthogonality). The family is drawn once per
        ``(low, high, levels)`` triple and cached.
        """
        if not isinstance(levels, int) or isinstance(levels, bool) or levels < 2:
            raise ValueError("levels must be an integer >= 2")
        if not (math.isfinite(low) and math.isfinite(high)) or not low < high:
            raise ValueError("low and high must be finite with low < high")
        key = (float(low), float(high), levels)
        if key not in self._level_memory:
            base = self.generate_random_vector()
            order = self._rng.permutation(self.dim)
            per_gap = (self.dim // 2) // (levels - 1)
            family = np.empty((levels, self.dim), dtype=np.uint8)
            family[0] = base
            for level in range(1, levels):
                vector = family[level - 1].copy()
                flips = order[(level - 1) * per_gap : level * per_gap]
                vector[flips] ^= 1
                family[level] = vector
            self._level_memory[key] = family
        return self._level_memory[key].copy()

    def encode_level(
        self, value: float, low: float, high: float, levels: int = 16
    ) -> np.ndarray[Any, Any]:
        """Encode a scalar as its nearest linear level hypervector.

        ``value`` is clipped into [low, high] and mapped to the closest
        of the ``levels`` cached level vectors for that range.
        """
        if not math.isfinite(value):
            raise ValueError("value must be finite")
        family = self.level_vectors(low, high, levels)
        clipped = min(max(value, low), high)
        index = round((clipped - low) / (high - low) * (levels - 1))
        encoded: np.ndarray[Any, Any] = family[index]
        return encoded


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
