# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — HDC encoder determinism, tie policies, level encoding

"""Seeded determinism, bundle tie policies, item memory, and level encoding."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
import pytest

from sc_neurocore.hdc import HDCEncoder


def test_seeded_encoders_reproduce_the_same_draws() -> None:
    first = HDCEncoder(dim=256, seed=7)
    second = HDCEncoder(dim=256, seed=7)
    for _ in range(3):
        assert np.array_equal(first.generate_random_vector(), second.generate_random_vector())


def test_different_seeds_draw_different_vectors() -> None:
    first = HDCEncoder(dim=256, seed=7)
    second = HDCEncoder(dim=256, seed=8)
    assert not np.array_equal(first.generate_random_vector(), second.generate_random_vector())


def test_item_memory_is_cached_and_name_distinct() -> None:
    enc = HDCEncoder(dim=256, seed=3)
    country = enc.item("country")
    assert np.array_equal(country, enc.item("country"))
    assert not np.array_equal(country, enc.item("capital"))
    country[0] ^= 1
    assert not np.array_equal(country, enc.item("country"))


def test_bundle_tie_policy_zeros_matches_historical_strict_majority() -> None:
    enc = HDCEncoder(dim=4)
    v1 = np.array([1, 0, 1, 0], dtype=np.uint8)
    v2 = np.array([1, 1, 0, 0], dtype=np.uint8)
    assert np.array_equal(enc.bundle([v1, v2]), np.array([1, 0, 0, 0], dtype=np.uint8))


def test_bundle_tie_policy_ones_sets_tied_positions() -> None:
    enc = HDCEncoder(dim=4, tie_policy="ones")
    v1 = np.array([1, 0, 1, 0], dtype=np.uint8)
    v2 = np.array([1, 1, 0, 0], dtype=np.uint8)
    assert np.array_equal(enc.bundle([v1, v2]), np.array([1, 1, 1, 0], dtype=np.uint8))


def test_bundle_tie_policy_random_is_seeded_and_majority_elsewhere() -> None:
    enc = HDCEncoder(dim=4, seed=11, tie_policy="random")
    reference = HDCEncoder(dim=4, seed=11)
    tie_break = reference.generate_random_vector()
    v1 = np.array([1, 0, 1, 0], dtype=np.uint8)
    v2 = np.array([1, 1, 0, 0], dtype=np.uint8)
    bundled = enc.bundle([v1, v2])
    assert bundled[0] == 1 and bundled[3] == 0
    assert bundled[1] == tie_break[1] and bundled[2] == tie_break[2]


def test_bundle_odd_count_never_consults_the_tie_policy() -> None:
    enc = HDCEncoder(dim=4, seed=11, tie_policy="random")
    v1 = np.array([1, 0, 1, 0], dtype=np.uint8)
    v2 = np.array([1, 1, 0, 0], dtype=np.uint8)
    v3 = np.array([1, 1, 0, 0], dtype=np.uint8)
    assert np.array_equal(enc.bundle([v1, v2, v3]), np.array([1, 1, 0, 0], dtype=np.uint8))


def test_bundle_without_ties_skips_the_tie_break_draw() -> None:
    enc = HDCEncoder(dim=2, seed=5, tie_policy="random")
    v1 = np.array([1, 0], dtype=np.uint8)
    v2 = np.array([1, 0], dtype=np.uint8)
    assert np.array_equal(enc.bundle([v1, v2]), np.array([1, 0], dtype=np.uint8))
    reference = HDCEncoder(dim=2, seed=5)
    assert np.array_equal(enc.generate_random_vector(), reference.generate_random_vector())


def test_level_encoding_distances_grow_linearly() -> None:
    enc = HDCEncoder(dim=1024, seed=2)
    family = enc.level_vectors(0.0, 1.0, 9)
    per_gap = (1024 // 2) // 8
    base = family[0]
    for level in range(9):
        distance = int(np.count_nonzero(np.bitwise_xor(base, family[level])))
        assert distance == level * per_gap


def test_level_encoding_is_cached_and_clips_out_of_range_values() -> None:
    enc = HDCEncoder(dim=512, seed=4)
    family = enc.level_vectors(-1.0, 1.0, 5)
    assert np.array_equal(enc.encode_level(-5.0, -1.0, 1.0, levels=5), family[0])
    assert np.array_equal(enc.encode_level(5.0, -1.0, 1.0, levels=5), family[4])
    assert np.array_equal(enc.encode_level(0.0, -1.0, 1.0, levels=5), family[2])
    assert np.array_equal(family, enc.level_vectors(-1.0, 1.0, 5))


@pytest.mark.parametrize("levels", (1, 0, -3))
def test_level_encoding_rejects_fewer_than_two_levels(levels: int) -> None:
    enc = HDCEncoder(dim=64, seed=1)
    with pytest.raises(ValueError, match="levels"):
        enc.level_vectors(0.0, 1.0, levels)


def test_level_encoding_rejects_non_integer_levels() -> None:
    enc = HDCEncoder(dim=64, seed=1)
    with pytest.raises(ValueError, match="levels"):
        enc.level_vectors(0.0, 1.0, cast(int, 4.0))


@pytest.mark.parametrize(("low", "high"), ((1.0, 0.0), (0.0, 0.0), (math.nan, 1.0)))
def test_level_encoding_rejects_invalid_ranges(low: float, high: float) -> None:
    enc = HDCEncoder(dim=64, seed=1)
    with pytest.raises(ValueError, match="low and high"):
        enc.level_vectors(low, high, 4)


def test_encode_level_rejects_non_finite_values() -> None:
    enc = HDCEncoder(dim=64, seed=1)
    with pytest.raises(ValueError, match="value"):
        enc.encode_level(math.nan, 0.0, 1.0)


@pytest.mark.parametrize("dim", (0, -5))
def test_encoder_rejects_non_positive_dimension(dim: int) -> None:
    with pytest.raises(ValueError, match="dim"):
        HDCEncoder(dim=dim)


def test_encoder_rejects_non_integer_dimension() -> None:
    with pytest.raises(ValueError, match="dim"):
        HDCEncoder(dim=cast(int, 64.0))


def test_encoder_rejects_unknown_tie_policy() -> None:
    with pytest.raises(ValueError, match="tie_policy"):
        HDCEncoder(dim=64, tie_policy="coin")


def test_majority_rejects_non_positive_count() -> None:
    enc = HDCEncoder(dim=4)
    with pytest.raises(ValueError, match="count"):
        enc.majority(np.zeros(4, dtype=np.int64), 0)
