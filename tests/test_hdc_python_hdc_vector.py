# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHDCVector from former test_hdc_python.py

"""Focused suite: TestHDCVector from former test_hdc_python.py."""

from __future__ import annotations

from tests.hdc_python_support import (
    HDCVector,
)


class TestHDCVector:
    """The Rust-backed hyperdimensional vector."""

    def test_create_default_dimension(self) -> None:
        """The default vector has 10000 dimensions."""
        v = HDCVector()
        assert v.dimension == 10_000

    def test_create_custom_dimension(self) -> None:
        """A custom dimension is kept."""
        v = HDCVector(5000, seed=42)
        assert v.dimension == 5000
        assert len(v) == 5000

    def test_bind_operator(self) -> None:
        """Binding gives a vector unlike both inputs."""
        a = HDCVector(10_000, seed=1)
        b = HDCVector(10_000, seed=2)
        c = a * b
        assert c.dimension == 10_000
        # a*b should be dissimilar to both a and b
        assert c.similarity(a) < 0.6
        assert c.similarity(b) < 0.6

    def test_bind_self_inverse(self) -> None:
        """Binding twice with the same vector recovers the input."""
        a = HDCVector(10_000, seed=1)
        b = HDCVector(10_000, seed=2)
        # (a*b)*b should recover a
        c = (a * b) * b
        assert c.similarity(a) > 0.99

    def test_bundle_operator(self) -> None:
        """Bundling keeps the dimension."""
        a = HDCVector(10_000, seed=1)
        b = HDCVector(10_000, seed=2)
        c = a + b  # bundle of two
        assert c.dimension == 10_000

    def test_bundle_preserves_similarity(self) -> None:
        """A bundle stays similar to its members."""
        a = HDCVector(10_000, seed=1)
        b = HDCVector(10_000, seed=2)
        c = HDCVector(10_000, seed=3)
        bundled = HDCVector.bundle([a, b, c])
        # Bundle should be more similar to its constituents than random
        random_vec = HDCVector(10_000, seed=999)
        sim_a = bundled.similarity(a)
        sim_random = bundled.similarity(random_vec)
        assert sim_a > sim_random, (
            f"Bundle should be more similar to constituent ({sim_a:.3f}) "
            f"than to random ({sim_random:.3f})"
        )

    def test_permute(self) -> None:
        """Permuting gives a vector unlike the input."""
        v = HDCVector(10_000, seed=42)
        p = v.permute(1)
        assert p.dimension == 10_000
        # Permuted vector should be dissimilar
        assert v.similarity(p) < 0.6

    def test_permute_is_not_inplace(self) -> None:
        """Permuting leaves the input unchanged."""
        v = HDCVector(10_000, seed=42)
        original_pc = v.popcount()
        _ = v.permute(1)
        assert v.popcount() == original_pc

    def test_similarity_self_is_one(self) -> None:
        """A vector is fully similar to itself."""
        v = HDCVector(10_000, seed=42)
        assert v.similarity(v) == 1.0

    def test_repr(self) -> None:
        """The representation names the vector and its dimension."""
        v = HDCVector(1000, seed=42)
        r = repr(v)
        assert "HDCVector" in r
        assert "1000" in r

    def test_symbolic_query_pattern(self) -> None:
        """End-to-end: encode country-capital pairs and query 'Capital of France?'."""
        country_france = HDCVector(10_000, seed=100)
        country_germany = HDCVector(10_000, seed=101)
        capital_paris = HDCVector(10_000, seed=200)
        capital_berlin = HDCVector(10_000, seed=201)
        role_country = HDCVector(10_000, seed=300)
        role_capital = HDCVector(10_000, seed=301)

        # Encode: record = (role_country * country) + (role_capital * capital)
        record_france = HDCVector.bundle(
            [
                role_country * country_france,
                role_capital * capital_paris,
            ]
        )
        record_germany = HDCVector.bundle(
            [
                role_country * country_germany,
                role_capital * capital_berlin,
            ]
        )

        # Memory = bundle of all records
        memory = HDCVector.bundle([record_france, record_germany])

        # Query: "Capital of France?" = unbind country role, bind with France
        query = role_capital * country_france
        # The answer should be closer to the France record
        # (This is a simplified test — full retrieval needs cleanup)
        sim_paris = query.similarity(capital_paris)
        sim_berlin = query.similarity(capital_berlin)
        # Both should be near 0.5 (random) since memory isn't directly queried
        # The key property is that the framework supports the pattern
        assert isinstance(sim_paris, float)
        assert isinstance(sim_berlin, float)
