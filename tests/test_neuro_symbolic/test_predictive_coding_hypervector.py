# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHypervector from former test_predictive_coding.py

"""Focused suite: TestHypervector from former test_predictive_coding.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.neuro_symbolic.predictive_coding import (
    _pack,
    _unpack,
    Hypervector,
    HYPERVECTOR_DIM,
)


class TestHypervector:
    """Construction, binding, permutation and bundling of hypervectors."""

    def test_zeros_popcount(self) -> None:
        """The zero hypervector has no set bits and the full dimension."""
        hv = Hypervector.zeros()
        assert hv.popcount() == 0
        assert hv.length == HYPERVECTOR_DIM

    def test_random_near_half_density(self) -> None:
        """A random hypervector has about half its bits set."""
        hv = Hypervector.random(0xDEAD)
        assert abs(hv.density() - 0.5) < 0.05

    def test_random_deterministic(self) -> None:
        """The same seed gives the same random hypervector."""
        a = Hypervector.random(42)
        b = Hypervector.random(42)
        assert np.array_equal(a.data, b.data)

    def test_random_different_seeds_orthogonal(self) -> None:
        """Different seeds give nearly orthogonal hypervectors."""
        a = Hypervector.random(1)
        b = Hypervector.random(2)
        assert abs(a.similarity(b)) < 0.1

    def test_bind_self_inverse(self) -> None:
        """Binding twice with the same vector recovers the original."""
        a = Hypervector.random(100)
        b = Hypervector.random(200)
        recovered = a.bind(b).bind(b)
        assert np.array_equal(a.data, recovered.data)

    def test_bind_dissimilar_to_inputs(self) -> None:
        """A bound vector is nearly orthogonal to both inputs."""
        a = Hypervector.random(10)
        b = Hypervector.random(20)
        c = a.bind(b)
        assert abs(c.similarity(a)) < 0.1
        assert abs(c.similarity(b)) < 0.1

    def test_permute_preserves_popcount(self) -> None:
        """Permutation keeps the number of set bits."""
        hv = Hypervector.random(333)
        permuted = hv.permute(7)
        assert permuted.popcount() == hv.popcount()

    def test_permute_changes_vector(self) -> None:
        """A one-step permutation changes the vector."""
        hv = Hypervector.random(555)
        permuted = hv.permute(1)
        assert not np.array_equal(hv.data, permuted.data)

    def test_threshold_bundle_majority(self) -> None:
        """A majority bundle of three vectors resembles each of them."""
        a = Hypervector.random(10)
        b = Hypervector.random(20)
        c = Hypervector.random(30)
        bundled = Hypervector.threshold_bundle([a, b, c])
        assert bundled.similarity(a) > 0.2
        assert bundled.similarity(b) > 0.2
        assert bundled.similarity(c) > 0.2

    def test_threshold_bundle_single(self) -> None:
        """Bundling one vector returns that vector."""
        a = Hypervector.random(42)
        bundled = Hypervector.threshold_bundle([a])
        assert np.array_equal(bundled.data, a.data)

    def test_hamming_self_zero(self) -> None:
        """A vector is at Hamming distance zero from itself."""
        a = Hypervector.random(77)
        assert a.hamming_distance(a) < 1e-10

    def test_similarity_self_one(self) -> None:
        """A vector has similarity one with itself."""
        a = Hypervector.random(88)
        assert abs(a.similarity(a) - 1.0) < 1e-10

    def test_pack_unpack_roundtrip(self) -> None:
        """Unpacking and repacking a hypervector gives it back."""
        hv = Hypervector.random(999)
        bits = _unpack(hv)
        repacked = _pack(bits, hv.length)
        assert np.array_equal(hv.data, repacked.data)
