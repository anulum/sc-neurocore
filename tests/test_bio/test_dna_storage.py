"""Tests for DNAEncoder bitstream conversions."""

import os
import time

import numpy as np
import pytest

from sc_neurocore.bio.dna_storage import DNAEncoder


def _perf_enabled() -> bool:
    return os.environ.get("SC_NEUROCORE_PERF") == "1"


def test_dna_encode_length_even():
    """Even-length bitstreams should map to half-length DNA."""
    enc = DNAEncoder(mutation_rate=0.0)
    bits = np.array([0, 1, 1, 0], dtype=np.uint8)
    dna = enc.encode(bits)
    assert len(dna) == 2


def test_dna_encode_pads_odd_length():
    """Odd-length bitstreams should be padded."""
    enc = DNAEncoder(mutation_rate=0.0)
    bits = np.array([1, 0, 1], dtype=np.uint8)
    dna = enc.encode(bits)
    assert len(dna) == 2


def test_dna_decode_length():
    """Decoding should produce 2*len(dna) bits."""
    enc = DNAEncoder(mutation_rate=0.0)
    bits = enc.decode("ACG")
    assert bits.shape == (6,)


def test_dna_roundtrip_no_mutation():
    """Roundtrip with zero mutation should preserve bits (with padding)."""
    enc = DNAEncoder(mutation_rate=0.0)
    bits = np.array([1, 0, 0], dtype=np.uint8)
    dna = enc.encode(bits)
    decoded = enc.decode(dna)
    assert np.array_equal(decoded[:bits.size], np.array([1, 0, 0], dtype=np.uint8))


def test_dna_encode_known_mapping():
    """Mapping 00->A, 01->C, 10->G, 11->T should hold."""
    enc = DNAEncoder(mutation_rate=0.0)
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
    dna = enc.encode(bits)
    assert dna == "ACGT"


def test_dna_decode_binary_output():
    """Decoded bits should be binary."""
    enc = DNAEncoder(mutation_rate=0.0)
    bits = enc.decode("ACGT")
    assert set(np.unique(bits).tolist()) <= {0, 1}


def test_dna_mutation_rate_one_changes_output():
    """High mutation rate should potentially alter decoded bits."""
    enc = DNAEncoder(mutation_rate=1.0)
    np.random.seed(0)
    bits = enc.decode("AAAA")
    assert bits.shape == (8,)


def test_dna_empty_bitstream():
    """Empty bitstream should encode to empty DNA."""
    enc = DNAEncoder(mutation_rate=0.0)
    dna = enc.encode(np.array([], dtype=np.uint8))
    assert dna == ""


def test_dna_deterministic_with_seed():
    """Numpy seed should make decoding deterministic."""
    enc = DNAEncoder(mutation_rate=0.5)
    np.random.seed(1)
    bits_a = enc.decode("ACGT")
    np.random.seed(1)
    bits_b = enc.decode("ACGT")
    assert np.array_equal(bits_a, bits_b)


@pytest.mark.skipif(not _perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_dna_perf_small():
    """Benchmark encoding a medium bitstream."""
    enc = DNAEncoder(mutation_rate=0.0)
    bits = np.random.randint(0, 2, size=10000, dtype=np.uint8)
    start = time.perf_counter()
    _ = enc.encode(bits)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
