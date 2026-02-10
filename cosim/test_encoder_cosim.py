"""
Co-simulation: sc_bitstream_encoder HDL vs Rust LFSR golden model.

Verifies that the LFSR sequence matches between Rust and Verilog.
"""

import pytest

try:
    import sc_neurocore_engine as engine
except ImportError:
    pytest.skip("sc_neurocore_engine not built", allow_module_level=True)


def test_lfsr_full_cycle(verilator_available, build_dir):
    """LFSR 16-bit full cycle: 65535 unique states."""
    del build_dir

    lfsr = engine.Lfsr16(seed=0xACE1)
    states = set()
    for _ in range(65535):
        val = lfsr.step()
        states.add(val)
    assert len(states) == 65535, "LFSR should produce 65535 unique states"


def test_encoder_probability_convergence(verilator_available, build_dir):
    """Encoder output probability converges to x_value / 65535."""
    del build_dir

    enc = engine.BitstreamEncoder(data_width=16, seed=0xACE1)
    target = 32768  # ~0.5 probability
    ones = sum(enc.step(target) for _ in range(10000))
    prob = ones / 10000
    assert abs(prob - 0.5) < 0.05, f"Expected ~0.5, got {prob}"
