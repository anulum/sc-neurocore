"""Strict blueprint semantics tests for LFSR + bitstream encoder."""

from sc_neurocore_engine import BitstreamEncoder, Lfsr16


def _lfsr_step(reg: int) -> int:
    feedback = ((reg >> 15) ^ (reg >> 13) ^ (reg >> 12) ^ (reg >> 10)) & 1
    return ((reg << 1) & 0xFFFF) | feedback


class TestLFSRBlueprintSemantics:
    def test_full_cycle_matches_blueprint_formula(self):
        reg = 0xACE1
        v3 = Lfsr16(seed=reg)

        for i in range(65535):
            reg = _lfsr_step(reg)
            assert v3.step() == reg, f"LFSR divergence at step {i}"

    def test_multiple_seeds(self):
        for seed in [0xACE1, 0xBEEF, 0xACE1 + 7, 0xBEEF + 13]:
            reg = seed
            v3 = Lfsr16(seed=seed)
            for i in range(1000):
                reg = _lfsr_step(reg)
                assert v3.step() == reg, f"Seed {seed:#06x} diverged at {i}"


class TestEncoderBlueprintSemantics:
    def test_step_then_compare_order(self):
        encoder = BitstreamEncoder(data_width=16, seed=0xACE1)
        x_value = 0xACE1
        reg = 0xACE1

        bits = []
        for _ in range(8):
            reg = _lfsr_step(reg)
            expected = 1 if reg < x_value else 0
            bits.append(encoder.step(x_value))
            assert bits[-1] == expected

        assert bits[0] == 1, "Strict step-then-compare should produce first bit = 1"
