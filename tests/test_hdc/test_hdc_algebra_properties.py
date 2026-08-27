# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — HDC algebraic contract properties

"""Randomised (seeded) algebraic laws from the HDC semantic contract."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.hdc import HDCEncoder

_SEEDS = (1, 17, 4093)
_DIMS = (33, 256, 1024)


def _pairs() -> list[tuple[int, int]]:
    return [(seed, dim) for seed in _SEEDS for dim in _DIMS]


@pytest.mark.parametrize(("seed", "dim"), _pairs())
def test_bind_is_self_inverse_and_commutative(seed: int, dim: int) -> None:
    enc = HDCEncoder(dim=dim, seed=seed)
    a = enc.generate_random_vector()
    b = enc.generate_random_vector()
    assert np.array_equal(enc.bind(a, a), np.zeros(dim, dtype=np.uint8))
    assert np.array_equal(enc.bind(enc.bind(a, b), b), a)
    assert np.array_equal(enc.bind(a, b), enc.bind(b, a))


@pytest.mark.parametrize(("seed", "dim"), _pairs())
def test_bind_preserves_hamming_distance(seed: int, dim: int) -> None:
    enc = HDCEncoder(dim=dim, seed=seed)
    a, b, c = (enc.generate_random_vector() for _ in range(3))
    direct = int(np.count_nonzero(np.bitwise_xor(a, b)))
    bound = int(np.count_nonzero(np.bitwise_xor(enc.bind(a, c), enc.bind(b, c))))
    assert bound == direct


@pytest.mark.parametrize(("seed", "dim"), _pairs())
def test_permute_composition_and_inverse(seed: int, dim: int) -> None:
    enc = HDCEncoder(dim=dim, seed=seed)
    v = enc.generate_random_vector()
    assert np.array_equal(enc.permute(enc.permute(v, 3), -3), v)
    assert np.array_equal(enc.permute(enc.permute(v, 2), 5), enc.permute(v, 7))


def test_permute_direction_is_a_right_rotation() -> None:
    enc = HDCEncoder(dim=4)
    v = np.array([1, 2, 3, 4], dtype=np.int32)
    assert np.array_equal(enc.permute(v, 1), np.array([4, 1, 2, 3], dtype=np.int32))


@pytest.mark.parametrize(
    ("seed", "dim"), [(seed, dim) for seed in _SEEDS for dim in _DIMS if dim >= 256]
)
def test_bundle_of_odd_count_preserves_similarity_to_members(seed: int, dim: int) -> None:
    """Statistical property: needs the quasi-orthogonal regime (dim >= 256)."""
    enc = HDCEncoder(dim=dim, seed=seed)
    members = [enc.generate_random_vector() for _ in range(5)]
    bundled = enc.bundle(members)
    stranger = enc.generate_random_vector()
    stranger_distance = int(np.count_nonzero(np.bitwise_xor(bundled, stranger)))
    for member in members:
        member_distance = int(np.count_nonzero(np.bitwise_xor(bundled, member)))
        assert member_distance < stranger_distance


@pytest.mark.parametrize(("seed", "dim"), _pairs())
def test_bundle_is_order_invariant(seed: int, dim: int) -> None:
    enc = HDCEncoder(dim=dim, seed=seed)
    members = [enc.generate_random_vector() for _ in range(5)]
    assert np.array_equal(enc.bundle(members), enc.bundle(list(reversed(members))))


@pytest.mark.parametrize(("seed", "dim"), _pairs())
def test_distance_is_symmetric_bounded_and_zero_on_equality(seed: int, dim: int) -> None:
    enc = HDCEncoder(dim=dim, seed=seed)
    a = enc.generate_random_vector()
    b = enc.generate_random_vector()
    ab = int(np.count_nonzero(np.bitwise_xor(a, b)))
    ba = int(np.count_nonzero(np.bitwise_xor(b, a)))
    assert ab == ba
    assert 0 <= ab <= dim
    assert int(np.count_nonzero(np.bitwise_xor(a, a))) == 0


@pytest.mark.parametrize("seed", _SEEDS)
def test_seeded_streams_reproduce_across_encoders(seed: int) -> None:
    first = HDCEncoder(dim=512, seed=seed)
    second = HDCEncoder(dim=512, seed=seed)
    for _ in range(5):
        assert np.array_equal(first.generate_random_vector(), second.generate_random_vector())
    assert np.array_equal(first.item("alpha"), second.item("alpha"))
    assert np.array_equal(first.level_vectors(0.0, 1.0, 4), second.level_vectors(0.0, 1.0, 4))


def test_malformed_construction_is_rejected() -> None:
    with pytest.raises(ValueError, match="dim"):
        HDCEncoder(dim=0)
    with pytest.raises(ValueError, match="tie_policy"):
        HDCEncoder(dim=8, tie_policy="maybe")
