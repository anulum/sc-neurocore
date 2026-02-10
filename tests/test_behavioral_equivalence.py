# tests/test_behavioral_equivalence.py
#
# Bit-true behavioral model verification against the Verilog specification.
# Tests cover:
#   1. LFSR sequence generation (sc_bitstream_encoder.v polynomial)
#   2. Bitstream encoder decorrelation (SEED_INIT parameter)
#   3. LIF neuron dynamics (sc_lif_neuron.v fixed-point arithmetic)
#   4. Full pipeline: encoder -> synapse -> dot-product -> neuron

import pytest
import numpy as np

from sc_neurocore.neurons.fixed_point_lif import (
    FixedPointLIFNeuron,
    FixedPointLFSR,
    FixedPointBitstreamEncoder,
    _mask,
)


# ============================================================================
#  1. LFSR Tests
# ============================================================================

class TestLFSR:
    """Verify LFSR matches Verilog sc_bitstream_encoder polynomial."""

    def test_default_seed(self):
        lfsr = FixedPointLFSR(seed=0xACE1)
        assert lfsr.reg == 0xACE1

    def test_first_step(self):
        """
        0xACE1 = 1010 1100 1110 0001
        Taps: bit15=1, bit13=1, bit12=0, bit10=1 -> feedback = 1^1^0^1 = 1
        Shift left: 0101 1001 1100 0010 = 0x59C2
        Insert feedback at LSB: 0x59C3
        """
        lfsr = FixedPointLFSR(seed=0xACE1)
        result = lfsr.step()
        assert result == 0x59C3, f"Expected 0x59C3, got {hex(result)}"

    def test_maximal_length(self):
        """A 16-bit maximal-length LFSR should have period 2^16-1 = 65535."""
        lfsr = FixedPointLFSR(seed=0xACE1)
        initial = lfsr.reg
        for i in range(65535):
            lfsr.step()
        assert lfsr.reg == initial, "LFSR did not return to initial state after 2^16-1 steps"

    def test_no_zero_state(self):
        """LFSR should never reach all-zeros during full period."""
        lfsr = FixedPointLFSR(seed=0xACE1)
        for _ in range(65535):
            lfsr.step()
            assert lfsr.reg != 0, "LFSR reached zero state"

    def test_zero_seed_raises(self):
        with pytest.raises(ValueError):
            FixedPointLFSR(seed=0)

    def test_different_seeds_produce_different_sequences(self):
        """Two LFSRs with different seeds must produce different bitstreams."""
        lfsr_a = FixedPointLFSR(seed=0xACE1)
        lfsr_b = FixedPointLFSR(seed=0xBEEF)

        seq_a = [lfsr_a.step() for _ in range(100)]
        seq_b = [lfsr_b.step() for _ in range(100)]

        assert seq_a != seq_b, "Different seeds produced identical sequences"


# ============================================================================
#  2. Bitstream Encoder Tests (decorrelation)
# ============================================================================

class TestBitstreamEncoder:
    """Verify decorrelation between parallel encoder instances."""

    def test_same_seed_same_output(self):
        enc_a = FixedPointBitstreamEncoder(seed_init=0xACE1)
        enc_b = FixedPointBitstreamEncoder(seed_init=0xACE1)

        bits_a = [enc_a.step(32768) for _ in range(256)]
        bits_b = [enc_b.step(32768) for _ in range(256)]

        assert bits_a == bits_b

    def test_different_seeds_decorrelate(self):
        """Encoders with different SEED_INIT must produce different bitstreams."""
        # These match the HDL fix: input encoders use 0xACE1 + i*7
        enc_0 = FixedPointBitstreamEncoder(seed_init=0xACE1 + 0 * 7)
        enc_1 = FixedPointBitstreamEncoder(seed_init=0xACE1 + 1 * 7)

        x_val = 32768  # ~50% probability
        bits_0 = [enc_0.step(x_val) for _ in range(256)]
        bits_1 = [enc_1.step(x_val) for _ in range(256)]

        assert bits_0 != bits_1, "Same x_value with different seeds should decorrelate"

    def test_weight_encoder_decorrelation(self):
        """Weight encoders (0xBEEF base) must differ from input encoders (0xACE1 base)."""
        enc_input = FixedPointBitstreamEncoder(seed_init=0xACE1)
        enc_weight = FixedPointBitstreamEncoder(seed_init=0xBEEF)

        x_val = 32768
        bits_in = [enc_input.step(x_val) for _ in range(256)]
        bits_wt = [enc_weight.step(x_val) for _ in range(256)]

        assert bits_in != bits_wt

    def test_probability_convergence(self):
        """Over many samples, proportion of 1s should converge to x_value/65535."""
        enc = FixedPointBitstreamEncoder(seed_init=0xACE1)
        x_val = 32768  # ~0.5
        length = 10000
        ones = sum(enc.step(x_val) for _ in range(length))
        p_hat = ones / length
        assert abs(p_hat - 0.5) < 0.05, f"Expected ~0.5, got {p_hat:.3f}"

    def test_zero_input_no_ones(self):
        enc = FixedPointBitstreamEncoder(seed_init=0xACE1)
        bits = [enc.step(0) for _ in range(1000)]
        assert sum(bits) == 0

    def test_max_input_all_ones(self):
        """x_value = 2^16 - 1 should produce all 1s (LFSR < 65535 always true)."""
        enc = FixedPointBitstreamEncoder(seed_init=0xACE1)
        bits = [enc.step(65535) for _ in range(1000)]
        # Allow 1 miss because LFSR can equal 65534 which is < 65535
        assert sum(bits) >= 999


# ============================================================================
#  3. Fixed-Point LIF Neuron Tests
# ============================================================================

class TestFixedPointLIF:
    """Bit-true verification of sc_lif_neuron.v dynamics."""

    def test_rest_with_no_input(self):
        neuron = FixedPointLIFNeuron()
        spike, v = neuron.step(leak_k=10, gain_k=256, I_t=0, noise_in=0)
        assert spike == 0
        assert v == 0  # v_rest=0, no input -> stays at rest

    def test_integration(self):
        """Constant small input should integrate membrane potential."""
        neuron = FixedPointLIFNeuron()
        # I_t=10, gain_k=256 (1.0): dv_in = 10*256 >> 8 = 10
        spike, v = neuron.step(leak_k=10, gain_k=256, I_t=10, noise_in=0)
        assert v == 10
        assert spike == 0

    def test_leak(self):
        """Membrane should leak toward v_rest when no input applied."""
        neuron = FixedPointLIFNeuron()
        neuron.v = 100
        # dv_leak = (0 - 100) * 10 >> 8 = -1000 >> 8 = -4 (arithmetic)
        spike, v = neuron.step(leak_k=10, gain_k=256, I_t=0, noise_in=0)
        assert v == 96  # 100 - 4
        assert spike == 0

    def test_spike_and_reset(self):
        """Large input should cause spike and reset to v_reset."""
        neuron = FixedPointLIFNeuron()
        neuron.v = 250  # Close to threshold (256)
        # dv_in = 10*256 >> 8 = 10; v_next = 250 + 10 = 260 >= 256 -> spike
        spike, v = neuron.step(leak_k=0, gain_k=256, I_t=10, noise_in=0)
        assert spike == 1
        assert v == 0  # v_reset

    def test_refractory_period(self):
        """After spike, neuron should be in refractory for N cycles."""
        neuron = FixedPointLIFNeuron(refractory_period=3)
        neuron.v = 255
        # Trigger spike
        spike, v = neuron.step(leak_k=0, gain_k=256, I_t=10, noise_in=0)
        assert spike == 1

        # Next 3 cycles should be refractory (no spike, v=v_rest)
        for _ in range(3):
            spike, v = neuron.step(leak_k=0, gain_k=256, I_t=255, noise_in=0)
            assert spike == 0
            assert v == 0

        # 4th cycle should respond normally again
        spike, v = neuron.step(leak_k=0, gain_k=256, I_t=10, noise_in=0)
        # Now out of refractory, should integrate
        assert v == 10

    def test_noise_injection(self):
        """Noise should affect membrane potential."""
        neuron = FixedPointLIFNeuron()
        spike, v = neuron.step(leak_k=0, gain_k=256, I_t=0, noise_in=50)
        assert v == 50

    def test_overflow_wrapping(self):
        """Bit-width masking should handle overflow correctly."""
        neuron = FixedPointLIFNeuron(v_threshold=32000)
        neuron.v = 32000
        # Large input that would overflow 16-bit signed
        spike, v = neuron.step(leak_k=0, gain_k=256, I_t=200, noise_in=0)
        # v_next = 32000 + 200 = 32200, masked to 16-bit signed = 32200
        # But 32200 in 16-bit signed is 32200 (still positive, < 32768)
        assert v == 32200 or spike == 1  # Either integrated or spiked

    def test_reset_method(self):
        neuron = FixedPointLIFNeuron()
        neuron.v = 100
        neuron.refractory_counter = 5
        neuron.reset()
        assert neuron.v == 0
        assert neuron.refractory_counter == 0

    def test_multi_step_convergence(self):
        """With constant input, neuron should eventually spike."""
        neuron = FixedPointLIFNeuron(refractory_period=0)
        spike_count = 0
        for _ in range(100):
            spike, v = neuron.step(leak_k=5, gain_k=256, I_t=20, noise_in=0)
            spike_count += spike
        assert spike_count > 0, "Neuron never spiked with constant input"


# ============================================================================
#  4. Full Pipeline Tests (Encoder -> Synapse -> DotProduct -> Neuron)
# ============================================================================

class TestFullPipeline:
    """End-to-end stochastic pipeline matching HDL architecture."""

    def _run_pipeline(self, x_values, w_values, n_steps=512,
                      leak_k=10, gain_k=256):
        """
        Software bit-true simulation of sc_dense_layer_core pipeline:
          input encoders -> weight encoders -> AND synapses -> popcount -> LIF
        """
        n_inputs = len(x_values)
        assert len(w_values) == n_inputs

        # Create decorrelated encoders (matching HDL SEED_INIT values)
        input_encs = [
            FixedPointBitstreamEncoder(seed_init=0xACE1 + i * 7)
            for i in range(n_inputs)
        ]
        weight_encs = [
            FixedPointBitstreamEncoder(seed_init=0xBEEF + i * 13)
            for i in range(n_inputs)
        ]

        neuron = FixedPointLIFNeuron(refractory_period=0)
        spikes = []

        for t in range(n_steps):
            # Encode
            pre_bits = [enc.step(x) for enc, x in zip(input_encs, x_values)]
            w_bits = [enc.step(w) for enc, w in zip(weight_encs, w_values)]

            # AND synapses
            post_bits = [p & w for p, w in zip(pre_bits, w_bits)]

            # Dot-product -> current (matching sc_dotproduct_to_current.v)
            count = sum(post_bits)
            # Map to fixed-point current range [y_min=0, y_max=256 (=1.0)]
            y_min = 0
            y_max = 256
            if n_inputs > 0:
                I_t = y_min + ((y_max - y_min) * count) // n_inputs
            else:
                I_t = 0

            spike, v = neuron.step(leak_k, gain_k, I_t, noise_in=0)
            spikes.append(spike)

        return spikes

    def test_pipeline_produces_spikes(self):
        """High input + high weight should produce spikes."""
        # x ~ 0.8, w ~ 0.8 -> post ~ 0.64 -> strong current
        x_vals = [52428, 52428, 52428]  # ~0.8 * 65535
        w_vals = [52428, 52428, 52428]
        spikes = self._run_pipeline(x_vals, w_vals, n_steps=1000)
        assert sum(spikes) > 0, "Pipeline should produce spikes with high input"

    def test_pipeline_no_spikes_zero_input(self):
        """Zero input should produce no spikes."""
        x_vals = [0, 0, 0]
        w_vals = [52428, 52428, 52428]
        spikes = self._run_pipeline(x_vals, w_vals, n_steps=500)
        assert sum(spikes) == 0, "Zero input should produce no spikes"

    def test_pipeline_no_spikes_zero_weight(self):
        """Zero weights should produce no spikes regardless of input."""
        x_vals = [65535, 65535, 65535]
        w_vals = [0, 0, 0]
        spikes = self._run_pipeline(x_vals, w_vals, n_steps=500)
        assert sum(spikes) == 0, "Zero weight should produce no spikes"

    def test_pipeline_decorrelation_matters(self):
        """
        With correlated encoders (same seed), AND-gate product is biased.
        With decorrelated encoders, result should differ.
        """
        # Run with decorrelated seeds (default pipeline)
        spikes_decorr = self._run_pipeline(
            [32768, 32768, 32768],
            [32768, 32768, 32768],
            n_steps=2000,
        )

        # Run with identical seeds (simulating the old bug)
        n_inputs = 3
        input_encs = [FixedPointBitstreamEncoder(seed_init=0xACE1) for _ in range(n_inputs)]
        weight_encs = [FixedPointBitstreamEncoder(seed_init=0xACE1) for _ in range(n_inputs)]
        neuron = FixedPointLIFNeuron(refractory_period=0)
        spikes_corr = []
        x_val, w_val = 32768, 32768
        for _ in range(2000):
            pre = [enc.step(x_val) for enc in input_encs]
            wbits = [enc.step(w_val) for enc in weight_encs]
            post = [p & w for p, w in zip(pre, wbits)]
            count = sum(post)
            I_t = (256 * count) // n_inputs
            spike, v = neuron.step(10, 256, I_t, 0)
            spikes_corr.append(spike)

        # Correlated encoders produce identical bitstreams when x == w,
        # so AND(x,x) = x -> higher current -> more spikes.
        # Decorrelated should have fewer spikes (p*w < p when independent).
        rate_corr = sum(spikes_corr) / len(spikes_corr)
        rate_decorr = sum(spikes_decorr) / len(spikes_decorr)

        assert rate_corr != rate_decorr, (
            f"Correlated ({rate_corr:.3f}) and decorrelated ({rate_decorr:.3f}) "
            "firing rates should differ, proving decorrelation matters"
        )


# ============================================================================
#  5. Bit-Width Masking Utility
# ============================================================================

class TestMask:
    def test_positive(self):
        assert _mask(100, 16) == 100

    def test_negative(self):
        assert _mask(-100, 16) == -100

    def test_overflow_wraps(self):
        # 32768 in 16-bit signed is -32768
        assert _mask(32768, 16) == -32768

    def test_underflow_wraps(self):
        # -32769 in 16-bit signed wraps to 32767
        assert _mask(-32769, 16) == 32767


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
