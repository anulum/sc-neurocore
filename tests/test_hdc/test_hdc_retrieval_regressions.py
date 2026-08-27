# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — HDC retrieval, collision, and dimension regressions

"""Seeded retrieval-accuracy, collision-rate, and dimension-sweep gates."""

from __future__ import annotations

from typing import Any

import numpy as np

from sc_neurocore.hdc import AssociativeMemory, HDCEncoder


def _noisy(
    rng: np.random.Generator, vector: np.ndarray[Any, Any], flip_fraction: float
) -> np.ndarray[Any, Any]:
    noisy = vector.copy()
    flips = rng.choice(vector.shape[0], size=int(vector.shape[0] * flip_fraction), replace=False)
    noisy[flips] ^= 1
    return noisy


def _retrieval_accuracy(dim: int, *, items: int, flip_fraction: float, seed: int) -> float:
    enc = HDCEncoder(dim=dim, seed=seed)
    memory = AssociativeMemory()
    stored = {f"item-{index}": enc.item(f"item-{index}") for index in range(items)}
    for label, vector in stored.items():
        memory.store(label, vector)
    rng = np.random.default_rng(seed + 1)
    hits = sum(
        1
        for label, vector in stored.items()
        if memory.query(_noisy(rng, vector, flip_fraction)) == label
    )
    return hits / items


def test_retrieval_survives_twenty_percent_noise_at_operating_dimension() -> None:
    assert _retrieval_accuracy(1024, items=50, flip_fraction=0.20, seed=5) == 1.0


def test_retrieval_accuracy_is_monotone_in_dimension() -> None:
    accuracies = [
        _retrieval_accuracy(dim, items=64, flip_fraction=0.35, seed=9) for dim in (64, 256, 2048)
    ]
    assert accuracies[0] <= accuracies[1] <= accuracies[2]
    assert accuracies[2] == 1.0


def test_random_vectors_concentrate_at_half_dimension_distance() -> None:
    """Quasi-orthogonality: pairwise distances stay inside a 4-sigma band."""
    dim = 4096
    for seed in (2, 23, 401):
        enc = HDCEncoder(dim=dim, seed=seed)
        vectors = [enc.generate_random_vector() for _ in range(12)]
        sigma = 0.5 * np.sqrt(dim)
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                distance = int(np.count_nonzero(np.bitwise_xor(vectors[i], vectors[j])))
                assert abs(distance - dim / 2) < 4.0 * sigma


def test_item_memory_produces_no_label_collisions() -> None:
    """Distinct names never map to identical vectors at operating dimension."""
    enc = HDCEncoder(dim=1024, seed=7)
    vectors = [enc.item(f"symbol-{index}") for index in range(200)]
    stacked = np.stack(vectors)
    unique_rows = np.unique(stacked, axis=0)
    assert unique_rows.shape[0] == len(vectors)


def test_cleanup_memory_tie_returns_earliest_stored_label() -> None:
    memory = AssociativeMemory()
    first = np.array([1, 1, 0, 0], dtype=np.uint8)
    second = np.array([0, 0, 1, 1], dtype=np.uint8)
    memory.store("first", first)
    memory.store("second", second)
    equidistant = np.array([1, 0, 1, 0], dtype=np.uint8)
    assert memory.query(equidistant) == "first"


def test_cleanup_memory_empty_returns_none() -> None:
    assert AssociativeMemory().query(np.zeros(8, dtype=np.uint8)) is None
