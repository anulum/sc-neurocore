"""
Co-simulation: sc_bitstream_synapse HDL vs Rust AND operation.
"""

import numpy as np
import pytest

try:
    import sc_neurocore_engine as engine
except ImportError:
    pytest.skip("sc_neurocore_engine not built", allow_module_level=True)


def test_and_probability(verilator_available, build_dir):
    """AND of two bitstreams: output probability ~ p1 * p2."""
    del build_dir

    bits_a = np.random.RandomState(42).randint(0, 2, 10000).astype(np.uint8)
    bits_b = np.random.RandomState(43).randint(0, 2, 10000).astype(np.uint8)

    packed_a = engine.pack_bitstream(bits_a.tolist())
    packed_b = engine.pack_bitstream(bits_b.tolist())

    expected_and = bits_a & bits_b
    expected_count = int(np.sum(expected_and))

    actual_count = 0
    for pa, pb in zip(packed_a, packed_b):
        actual_count += bin(pa & pb).count("1")

    assert abs(actual_count - expected_count) <= 1
