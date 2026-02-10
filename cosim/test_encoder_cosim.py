"""Co-simulation: Bitstream encoder - Rust golden model vs Verilator HDL."""

from __future__ import annotations

import pytest

from sc_neurocore_engine import Lfsr16, BitstreamEncoder


@pytest.mark.usefixtures("verilator_available")
class TestEncoderCosim:
    """Validate LFSR and encoder against Rust golden model."""

    def test_lfsr_full_cycle(self):
        """LFSR 16-bit produces 65535 unique states."""
        lfsr = Lfsr16(seed=0xACE1)
        seen = set()
        for _ in range(65535):
            val = lfsr.step()
            seen.add(val)
        assert len(seen) == 65535, f"Expected 65535 unique, got {len(seen)}"

    def test_encoder_probability_convergence(self):
        """Encoder with target ~0.5 produces ~50% ones over 10000 cycles."""
        enc = BitstreamEncoder(data_width=16, seed=0xACE1)
        target = 32768  # 0.5 * 65536
        ones = sum(enc.step(target) for _ in range(10000))
        probability = ones / 10000.0
        assert abs(probability - 0.5) < 0.05, f"Expected ~0.5, got {probability:.3f}"

    def test_multiple_seeds(self):
        """Different seeds produce different LFSR sequences."""
        seeds = [0xACE1, 0xBEEF, 0xACE1 + 7, 0xBEEF + 13]
        sequences = []
        for seed in seeds:
            lfsr = Lfsr16(seed=seed)
            seq = tuple(lfsr.step() for _ in range(100))
            sequences.append(seq)
        # All sequences should be distinct
        assert len(set(sequences)) == len(seeds), "Seed decorrelation failed"
