# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for tinysc_riscv ports (bitstream, lfsr, neuron, network, telemetry, weights)

"""Comprehensive tests mirroring the Rust test suites from tinysc_riscv."""

import pytest
from sc_neurocore.edge.bitstream import (
    popcount32,
    popcount_slice,
    sc_and,
    sc_or,
    sc_xor,
    sc_sub,
    sc_mux,
    and_packed,
    mux_packed,
    probability,
    scc,
    MASK32,
)
from sc_neurocore.edge.lfsr import Lfsr16
from sc_neurocore.edge.neuron import LifNeuron, IzhikevichNeuron
from sc_neurocore.edge.sc_network import SCLayer, SCNetwork
from sc_neurocore.edge.telemetry import TelemetryRing, DeviceTelemetry
from sc_neurocore.edge.weights import (
    serialize_weights,
    deserialize_weights,
    WeightHeader,
    WEIGHT_MAGIC,
)


# ===== Bitstream Tests (mirror bitstream.rs tests) =====


class TestPopcount:
    def test_zero(self):
        assert popcount32(0) == 0

    def test_all_ones(self):
        assert popcount32(MASK32) == 32

    def test_alternating(self):
        assert popcount32(0xAAAA_AAAA) == 16

    def test_single_bit(self):
        for i in range(32):
            assert popcount32(1 << i) == 1

    def test_slice(self):
        assert popcount_slice([MASK32, MASK32]) == 64

    def test_slice_empty(self):
        assert popcount_slice([]) == 0


class TestSCOps:
    def test_sc_and(self):
        assert sc_and(0b1010, 0b1100) == 0b1000

    def test_sc_or(self):
        assert sc_or(0b1010, 0b0101) == 0b1111

    def test_sc_xor(self):
        assert sc_xor(0b1010, 0b1100) == 0b0110

    def test_sc_sub(self):
        assert sc_sub(0b1110, 0b0110) == 0b1000

    def test_sc_mux(self):
        assert sc_mux(0xFF, 0x00, 0x0F) == 0x0F

    def test_and_packed(self):
        a = [0xAAAA_AAAA, 0xFFFF_FFFF]
        b = [0x5555_5555, 0x0000_FFFF]
        out = and_packed(a, b)
        assert out[0] == 0
        assert out[1] == 0x0000_FFFF

    def test_mux_packed(self):
        a = [0xFFFF_FFFF]
        b = [0x0000_0000]
        s = [0x0000_FFFF]
        out = mux_packed(a, b, s)
        assert out[0] == 0x0000_FFFF

    def test_and_packed_rejects_length_mismatch(self):
        with pytest.raises(AssertionError):
            and_packed([0x1, 0x2], [0x1])

    def test_mux_packed_rejects_length_mismatch(self):
        with pytest.raises(AssertionError):
            mux_packed([0x1], [0x0, 0x1], [0x1])


class TestProbability:
    def test_all_ones(self):
        assert abs(probability([MASK32], 32) - 1.0) < 1e-6

    def test_all_zeros(self):
        assert abs(probability([0], 32) - 0.0) < 1e-6

    def test_half(self):
        assert abs(probability([0xAAAA_AAAA], 32) - 0.5) < 1e-6

    def test_zero_length(self):
        assert probability([MASK32], 0) == 0.0


class TestSCC:
    def test_identical(self):
        a = [0xAAAA_AAAA]
        assert abs(scc(a, a, 32) - 1.0) < 0.01

    def test_anticorrelated(self):
        a = [0xAAAA_AAAA]
        b = [0x5555_5555]
        assert abs(scc(a, b, 32) - (-1.0)) < 0.01

    def test_zero_length(self):
        assert scc([0], [0], 0) == 0.0

    def test_zero_density_streams_hit_numerator_floor(self):
        # Empty (all-zero) streams over a non-zero length give pa=pb=p_and=0,
        # so the numerator collapses to the |num|<eps floor: the coefficient
        # is defined as 0 rather than 0/0.
        assert scc([0x0000_0000], [0x0000_0000], 32) == 0.0

    def test_under_counted_length_hits_denominator_floor(self):
        # bit_length under-counts the bits packed into the words, pushing pa
        # above 1 and breaking the p_and<=min(pa,pb) invariant: here pa=2, pb=1
        # make the denominator collapse to 0 while the numerator stays nonzero,
        # exercising the |denom|<eps floor that keeps the result finite.
        assert scc([0b11], [0b01], 1) == 0.0


# ===== LFSR Tests =====


class TestLfsr16:
    def test_nonzero_seed(self):
        lfsr = Lfsr16(0xACE1)
        assert lfsr.reg != 0

    def test_step_changes_state(self):
        lfsr = Lfsr16(0xACE1)
        s0 = lfsr.reg
        lfsr.step()
        assert lfsr.reg != s0

    def test_encode_length(self):
        lfsr = Lfsr16(0xACE1)
        words = lfsr.encode(32768, 1024)
        assert len(words) == 32  # 1024 / 32 = 32 words

    def test_encode_float_half(self):
        lfsr = Lfsr16(0xACE1)
        words = lfsr.encode_float(0.5, 1024)
        pc = popcount_slice(words)
        assert 400 < pc < 600  # ~50% ± tolerance

    def test_zero_seed_uses_default(self):
        lfsr = Lfsr16(0)
        assert lfsr.reg == 0xACE1

    def test_period_uniqueness(self):
        lfsr = Lfsr16(0xACE1)
        seen = set()
        for _ in range(1000):
            seen.add(lfsr.step())
        assert len(seen) == 1000  # should all be unique in first 1000


# ===== Neuron Tests =====


class TestLifNeuron:
    def test_quiescent(self):
        n = LifNeuron(threshold=100)
        assert not n.tick([0])
        assert n.membrane == 0

    def test_excitation(self):
        n = LifNeuron(threshold=10, leak_shift=8)
        for _ in range(20):
            n.tick([MASK32])
        assert n.spike_count > 0

    def test_reset(self):
        n = LifNeuron()
        n.membrane = 999
        n.spike_count = 5
        n.reset()
        assert n.membrane == 0
        assert n.spike_count == 0


class TestIzhikevichNeuron:
    def test_regular_spiking(self):
        n = IzhikevichNeuron.regular_spiking()
        assert n.a_q16 == 1311

    def test_fast_spiking(self):
        n = IzhikevichNeuron.fast_spiking()
        assert n.a_q16 == 6554

    def test_modes_differ(self):
        rs = IzhikevichNeuron.regular_spiking()
        fs = IzhikevichNeuron.fast_spiking()
        assert rs.a_q16 != fs.a_q16

    def test_tick_without_strong_input_does_not_immediately_spike(self):
        n = IzhikevichNeuron.regular_spiking()
        spiked = n.tick(0)
        assert spiked is False
        assert n.spike_count == 0


# ===== Network Tests =====


class TestSCNetwork:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"n_inputs": 0, "n_outputs": 1},
            {"n_inputs": 1, "n_outputs": 0},
            {"n_inputs": 1, "n_outputs": 1, "threshold": -1},
            {"n_inputs": 65, "n_outputs": 1},
            {"n_inputs": 1, "n_outputs": 65},
            {"n_inputs": 1, "n_outputs": 1, "sc_mode": "bipolar"},
        ],
    )
    def test_layer_invalid_configuration(self, kwargs):
        with pytest.raises(ValueError):
            SCLayer(**kwargs)

    def test_layer_invalid_weight_shape(self):
        with pytest.raises(ValueError, match="one row"):
            SCLayer(n_inputs=1, n_outputs=2, weights=[[0xFFFF_FFFF]])

    def test_layer_rejects_invalid_weight_word(self):
        with pytest.raises(ValueError, match="unsigned"):
            SCLayer(n_inputs=1, n_outputs=1, weights=[[MASK32 + 1]])

    def test_layer_rejects_invalid_input_words(self):
        layer = SCLayer(n_inputs=4, n_outputs=1)
        with pytest.raises(ValueError, match="input_words"):
            layer.forward([], 32)
        with pytest.raises(ValueError, match="unsigned"):
            layer.forward([-1], 32)
        with pytest.raises(ValueError, match="bit_length"):
            layer.forward([0], 0)

    def test_network_invalid_configuration(self):
        with pytest.raises(ValueError, match="bit_length"):
            SCNetwork(bit_length=0)
        with pytest.raises(ValueError, match="unipolar"):
            SCNetwork(sc_mode="bipolar")

    def test_network_rejects_invalid_probabilities(self):
        net = SCNetwork(bit_length=256)
        net.add_layer(SCLayer(n_inputs=2, n_outputs=1))
        with pytest.raises(ValueError, match="probabilities"):
            net.run([0.5, 1.1])

    def test_network_rejects_input_length_mismatch(self):
        net = SCNetwork(bit_length=256)
        net.add_layer(SCLayer(n_inputs=2, n_outputs=1))
        with pytest.raises(ValueError, match="n_inputs"):
            net.run([0.5])

    def test_add_layer_rejects_sc_mode_mismatch(self):
        net = SCNetwork(bit_length=256, sc_mode="unipolar")
        layer = SCLayer(n_inputs=1, n_outputs=1, sc_mode="unipolar")
        layer.sc_mode = "bipolar"  # force mismatch branch post-init
        with pytest.raises(ValueError, match="sc_mode"):
            net.add_layer(layer)

    def test_empty_network(self):
        net = SCNetwork(bit_length=256)
        assert net.run([]) == []

    def test_single_layer(self):
        net = SCNetwork(bit_length=256)
        net.add_layer(SCLayer(n_inputs=4, n_outputs=2, threshold=1))
        result = net.run([0.9, 0.9, 0.9, 0.9])
        assert len(result) == 2

    def test_layer_count(self):
        net = SCNetwork()
        net.add_layer(SCLayer(n_inputs=4, n_outputs=2))
        net.add_layer(SCLayer(n_inputs=2, n_outputs=1))
        assert net.layer_count == 2
        assert net.total_neurons == 3

    def test_from_weights_roundtrip(self):
        class LayerHeader:
            def __init__(self, n_inputs: int, n_outputs: int, threshold: int) -> None:
                self.n_inputs = n_inputs
                self.n_outputs = n_outputs
                self.threshold = threshold

        source = SCNetwork(bit_length=256)
        source.add_layer(SCLayer(n_inputs=2, n_outputs=1, threshold=2, weights=[[0xFFFF_FFFF]]))
        exported = source.export_weights()
        layers_data = [
            (LayerHeader(n_inputs, n_outputs, threshold), rows)
            for n_inputs, n_outputs, threshold, rows in exported
        ]
        restored = SCNetwork.from_weights(layers_data, bit_length=256, lfsr_seed=0x1234)
        assert restored.layer_count == 1
        assert restored.layers[0].weights == [[0xFFFF_FFFF]]
        assert restored.lfsr_seed == 0x1234


# ===== Telemetry Tests =====


class TestTelemetryRing:
    def test_empty_ring_defaults(self):
        ring = TelemetryRing(0)
        assert ring.capacity == 1
        assert ring.count == 0
        assert ring.mean() == 0.0
        assert ring.last() == 0

    def test_push_and_mean(self):
        ring = TelemetryRing(4)
        for v in [10, 20, 30, 40]:
            ring.push(v)
        assert ring.mean() == 25.0

    def test_last(self):
        ring = TelemetryRing(4)
        ring.push(42)
        assert ring.last() == 42

    def test_overflow(self):
        ring = TelemetryRing(2)
        for v in [1, 2, 3, 4, 5]:
            ring.push(v)
        assert ring.count == 2
        assert ring.last() == 5


class TestDeviceTelemetry:
    def test_record(self):
        dt = DeviceTelemetry()
        dt.record("L0", 5, 16)
        dt.record("L0", 3, 16)
        assert dt.total_ticks == 2
        assert dt.total_spikes == 8
        layer = dt.get_layer("L0")
        assert layer.tick_count == 2

    def test_summary(self):
        dt = DeviceTelemetry()
        dt.record("L0", 10, 32)
        s = dt.summary()
        assert s["total_spikes"] == 10
        assert "L0" in s["layers"]

    def test_layer_rate_and_zero_neuron_utilization_path(self):
        dt = DeviceTelemetry()
        dt.record("L0", 6, 0)  # should not push utilization sample
        dt.record("L0", 2, 10)  # should push one utilization sample (20%)
        layer = dt.get_layer("L0")
        assert layer.lifetime_spike_rate == pytest.approx(4.0)
        assert layer.mean_spike_rate == pytest.approx(4.0)
        assert layer.mean_utilization == pytest.approx(20.0)

    def test_get_layer_is_idempotent_and_initialises_zero_rates(self):
        dt = DeviceTelemetry()
        first = dt.get_layer("L-new")
        second = dt.get_layer("L-new")
        assert first is second
        assert first.lifetime_spike_rate == 0.0
        assert first.mean_spike_rate == 0.0
        assert first.mean_utilization == 0.0


# ===== Weights Tests =====


class TestWeights:
    def test_header_roundtrip(self):
        h = WeightHeader(n_layers=3)
        data = h.to_bytes()
        h2 = WeightHeader.from_bytes(data)
        assert h2.magic == WEIGHT_MAGIC
        assert h2.n_layers == 3

    def test_validate(self):
        h = WeightHeader()
        assert h.validate()
        h.magic = 0xDEAD
        assert not h.validate()

    def test_serialize_roundtrip(self):
        weights = [
            (4, 2, 512, [[0xAAAA_AAAA], [0x5555_5555]]),
        ]
        blob = serialize_weights(weights)
        layers = deserialize_weights(blob)
        assert len(layers) == 1
        lh, rows = layers[0]
        assert lh.n_inputs == 4
        assert lh.n_outputs == 2
        assert rows[0] == [0xAAAA_AAAA]
        assert rows[1] == [0x5555_5555]

    def test_multi_layer(self):
        weights = [
            (32, 4, 256, [[0xFF] * 1 for _ in range(4)]),
            (4, 2, 128, [[0x0F] * 1 for _ in range(2)]),
        ]
        blob = serialize_weights(weights)
        layers = deserialize_weights(blob)
        assert len(layers) == 2

    def test_invalid_magic_raises(self):
        data = b"\x00" * 16
        with pytest.raises(ValueError, match="Invalid weight blob"):
            deserialize_weights(data)


# ===== Integration Tests =====


class TestCascadeSemantics:
    def test_two_layer_cascade_output_size(self):
        net = SCNetwork(bit_length=256)
        net.add_layer(SCLayer(n_inputs=4, n_outputs=4, threshold=1))
        net.add_layer(SCLayer(n_inputs=4, n_outputs=2, threshold=1))
        result = net.run([0.9, 0.9, 0.9, 0.9])
        assert len(result) == 2

    def test_cascade_deterministic(self):
        net = SCNetwork(bit_length=256)
        net.add_layer(SCLayer(n_inputs=4, n_outputs=4, threshold=1))
        net.add_layer(SCLayer(n_inputs=4, n_outputs=2, threshold=1))
        r1 = net.run([0.5, 0.5, 0.5, 0.5])
        r2 = net.run([0.5, 0.5, 0.5, 0.5])
        assert r1 == r2

    def test_three_layer_cascade(self):
        net = SCNetwork(bit_length=256)
        net.add_layer(SCLayer(n_inputs=8, n_outputs=4, threshold=1))
        net.add_layer(SCLayer(n_inputs=4, n_outputs=4, threshold=1))
        net.add_layer(SCLayer(n_inputs=4, n_outputs=1, threshold=1))
        result = net.run([0.9] * 8)
        assert len(result) == 1


class TestWeightNetworkIntegration:
    def test_export_import_roundtrip(self):
        net = SCNetwork(bit_length=256)
        net.add_layer(SCLayer(n_inputs=4, n_outputs=2, threshold=1))
        r1 = net.run([0.9, 0.9, 0.9, 0.9])

        blob = serialize_weights(net.export_weights())
        loaded = deserialize_weights(blob)
        net2 = SCNetwork.from_weights(loaded, bit_length=256)
        r2 = net2.run([0.9, 0.9, 0.9, 0.9])
        assert r1 == r2

    def test_export_preserves_structure(self):
        net = SCNetwork(bit_length=512)
        net.add_layer(SCLayer(n_inputs=32, n_outputs=8, threshold=100))
        net.add_layer(SCLayer(n_inputs=8, n_outputs=2, threshold=50))
        exported = net.export_weights()
        assert len(exported) == 2
        assert exported[0][0] == 32  # n_inputs
        assert exported[0][1] == 8  # n_outputs
        assert exported[1][2] == 50  # threshold


class TestEndToEnd:
    def test_inference_telemetry_pipeline(self):
        net = SCNetwork(bit_length=256)
        net.add_layer(SCLayer(n_inputs=4, n_outputs=2, threshold=1))
        dt = DeviceTelemetry()
        for _ in range(10):
            spikes = net.run([0.5, 0.5, 0.5, 0.5])
            dt.record("output", sum(spikes), len(spikes))
        s = dt.summary()
        assert s["total_ticks"] == 10
        assert "output" in s["layers"]

    def test_lfsr_encode_scc_consistency(self):
        lfsr1 = Lfsr16(0xACE1)
        lfsr2 = Lfsr16(0xACE1)
        a = lfsr1.encode_float(0.5, 1024)
        b = lfsr2.encode_float(0.5, 1024)
        corr = scc(a, b, 1024)
        assert abs(corr - 1.0) < 0.01

    def test_scc_uncorrelated(self):
        a = Lfsr16(0xACE1).encode_float(0.5, 1024)
        b = Lfsr16(0x1234).encode_float(0.5, 1024)
        corr = scc(a, b, 1024)
        assert abs(corr) < 0.15


def test_sclayer_rejects_mismatched_weight_row_length() -> None:
    # Right number of rows but a row whose word count does not match
    # words_per_input is rejected by the SCLayer weight validator.
    with pytest.raises(ValueError, match="each weight row must match words_per_input"):
        SCLayer(n_inputs=64, n_outputs=2, weights=[[0], [0, 0, 0]])
