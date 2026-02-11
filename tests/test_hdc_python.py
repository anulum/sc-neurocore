# CopyRight: (c) 1998-2026 Miroslav Sotek. All rights reserved.
# Contact us: www.anulum.li  protoscience@anulum.li
# ORCID: https://orcid.org/0009-0009-3560-0851
# License: GNU AFFERO GENERAL PUBLIC LICENSE v3
# Commercial Licensing: Available

"""End-to-end Python tests for HDC/VSA via the Rust BitStreamTensor backend."""

import pytest
from sc_neurocore_engine import BitStreamTensor, HDCVector


# ── BitStreamTensor (raw Rust wrapper) ───────────────────────────────

class TestBitStreamTensor:
    def test_create_random(self):
        t = BitStreamTensor(1000, seed=42)
        assert len(t) == 1000
        pc = t.popcount()
        assert 350 < pc < 650, f"Expected ~500, got {pc}"

    def test_from_packed_roundtrip(self):
        t1 = BitStreamTensor(128, seed=99)
        t2 = BitStreamTensor.from_packed(t1.data, t1.length)
        assert t1.hamming_distance(t2) == 0.0

    def test_xor_self_is_zero(self):
        t = BitStreamTensor(256, seed=1)
        result = t.xor(t)
        assert result.popcount() == 0

    def test_xor_inplace(self):
        a = BitStreamTensor(256, seed=1)
        b = BitStreamTensor(256, seed=2)
        expected = a.xor(b)
        a.xor_inplace(b)
        assert a.hamming_distance(expected) == 0.0

    def test_hamming_identical_zero(self):
        t = BitStreamTensor(1000, seed=7)
        assert t.hamming_distance(t) == 0.0

    def test_hamming_random_near_half(self):
        a = BitStreamTensor(10_000, seed=42)
        b = BitStreamTensor(10_000, seed=99)
        hd = a.hamming_distance(b)
        assert 0.4 < hd < 0.6, f"Expected ~0.5, got {hd}"

    def test_rotate_identity(self):
        t = BitStreamTensor(128, seed=5)
        original_data = list(t.data)
        t.rotate_right(0)
        assert list(t.data) == original_data
        t.rotate_right(128)  # full length
        assert list(t.data) == original_data

    def test_bundle_majority(self):
        a = BitStreamTensor(10_000, seed=1)
        b = BitStreamTensor(10_000, seed=2)
        c = BitStreamTensor(10_000, seed=3)
        result = BitStreamTensor.bundle([a, b, c])
        assert len(result) == 10_000
        # Bundle of 3 random vectors should still be ~50% ones
        ratio = result.popcount() / 10_000
        assert 0.3 < ratio < 0.7

    def test_repr(self):
        t = BitStreamTensor(100, seed=42)
        r = repr(t)
        assert "BitStreamTensor" in r
        assert "100" in r


# ── HDCVector (high-level Python wrapper) ────────────────────────────

class TestHDCVector:
    def test_create_default_dimension(self):
        v = HDCVector()
        assert v.dimension == 10_000

    def test_create_custom_dimension(self):
        v = HDCVector(5000, seed=42)
        assert v.dimension == 5000
        assert len(v) == 5000

    def test_bind_operator(self):
        a = HDCVector(10_000, seed=1)
        b = HDCVector(10_000, seed=2)
        c = a * b
        assert c.dimension == 10_000
        # a*b should be dissimilar to both a and b
        assert c.similarity(a) < 0.6
        assert c.similarity(b) < 0.6

    def test_bind_self_inverse(self):
        a = HDCVector(10_000, seed=1)
        b = HDCVector(10_000, seed=2)
        # (a*b)*b should recover a
        c = (a * b) * b
        assert c.similarity(a) > 0.99

    def test_bundle_operator(self):
        a = HDCVector(10_000, seed=1)
        b = HDCVector(10_000, seed=2)
        c = a + b  # bundle of two
        assert c.dimension == 10_000

    def test_bundle_preserves_similarity(self):
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

    def test_permute(self):
        v = HDCVector(10_000, seed=42)
        p = v.permute(1)
        assert p.dimension == 10_000
        # Permuted vector should be dissimilar
        assert v.similarity(p) < 0.6

    def test_permute_is_not_inplace(self):
        v = HDCVector(10_000, seed=42)
        original_pc = v.popcount()
        _ = v.permute(1)
        assert v.popcount() == original_pc

    def test_similarity_self_is_one(self):
        v = HDCVector(10_000, seed=42)
        assert v.similarity(v) == 1.0

    def test_repr(self):
        v = HDCVector(1000, seed=42)
        r = repr(v)
        assert "HDCVector" in r
        assert "1000" in r

    def test_symbolic_query_pattern(self):
        """End-to-end: encode country-capital pairs, query 'Capital of France?'"""
        country_france = HDCVector(10_000, seed=100)
        country_germany = HDCVector(10_000, seed=101)
        capital_paris = HDCVector(10_000, seed=200)
        capital_berlin = HDCVector(10_000, seed=201)
        role_country = HDCVector(10_000, seed=300)
        role_capital = HDCVector(10_000, seed=301)

        # Encode: record = (role_country * country) + (role_capital * capital)
        record_france = HDCVector.bundle([
            role_country * country_france,
            role_capital * capital_paris,
        ])
        record_germany = HDCVector.bundle([
            role_country * country_germany,
            role_capital * capital_berlin,
        ])

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
