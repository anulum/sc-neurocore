"""Tests for hyperdimensional computing base components."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.hdc.base import HDCEncoder, AssociativeMemory


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_hdc_generate_random_vector_shape_and_values():
    """Random vectors should match dim and be binary."""
    enc = HDCEncoder(dim=16)
    vec = enc.generate_random_vector()
    assert vec.shape == (16,)
    assert set(np.unique(vec).tolist()) <= {0, 1}


def test_hdc_bind_xor():
    """Bind should XOR two vectors."""
    enc = HDCEncoder(dim=4)
    v1 = np.array([1, 0, 1, 0], dtype=np.uint8)
    v2 = np.array([1, 1, 0, 0], dtype=np.uint8)
    out = enc.bind(v1, v2)
    assert np.array_equal(out, np.array([0, 1, 1, 0], dtype=np.uint8))


def test_hdc_bundle_empty_returns_zeros():
    """Bundling empty list returns zeros."""
    enc = HDCEncoder(dim=8)
    out = enc.bundle([])
    assert np.array_equal(out, np.zeros(8, dtype=np.uint8))


def test_hdc_bundle_majority_for_two_vectors():
    """Bundling two vectors acts like AND (strict majority)."""
    enc = HDCEncoder(dim=4)
    v1 = np.array([1, 0, 1, 0], dtype=np.uint8)
    v2 = np.array([1, 1, 0, 0], dtype=np.uint8)
    out = enc.bundle([v1, v2])
    assert np.array_equal(out, np.array([1, 0, 0, 0], dtype=np.uint8))


def test_hdc_permute_shift():
    """Permute should roll vector by shifts."""
    enc = HDCEncoder(dim=4)
    v = np.array([1, 2, 3, 4], dtype=np.int32)
    out = enc.permute(v, shifts=1)
    assert np.array_equal(out, np.array([4, 1, 2, 3], dtype=np.int32))


def test_assoc_memory_store_and_query_exact():
    """Querying an exact vector should return its label."""
    mem = AssociativeMemory()
    v = np.array([1, 0, 1, 0], dtype=np.uint8)
    mem.store("a", v)
    assert mem.query(v) == "a"


def test_assoc_memory_query_closest():
    """Query returns label with smallest Hamming distance."""
    mem = AssociativeMemory()
    mem.store("a", np.array([1, 1, 1, 1], dtype=np.uint8))
    mem.store("b", np.array([0, 0, 0, 0], dtype=np.uint8))
    query = np.array([1, 1, 0, 0], dtype=np.uint8)
    assert mem.query(query) == "a"


def test_assoc_memory_empty_query_returns_none():
    """Querying empty memory should return None."""
    mem = AssociativeMemory()
    query = np.array([1, 0, 1, 0], dtype=np.uint8)
    assert mem.query(query) is None


def test_assoc_memory_overwrite_label():
    """Storing the same label should overwrite the vector."""
    mem = AssociativeMemory()
    mem.store("a", np.array([1, 1, 1], dtype=np.uint8))
    mem.store("a", np.array([0, 0, 0], dtype=np.uint8))
    assert np.array_equal(mem.memory["a"], np.array([0, 0, 0], dtype=np.uint8))


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_hdc_perf_bundle():
    """Benchmark bundling many vectors."""
    enc = HDCEncoder(dim=1024)
    vectors = [enc.generate_random_vector() for _ in range(50)]
    start = time.perf_counter()
    _ = enc.bundle(vectors)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
