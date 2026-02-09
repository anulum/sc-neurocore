"""Tests to fill coverage gaps in partially-covered modules."""

import os
import tempfile
import numpy as np
import pytest


# ── neurons ──────────────────────────────────────────────────────────
class TestSCIzhikevichNeuron:
    def test_construction_and_reset(self):
        from sc_neurocore.neurons.sc_izhikevich import SCIzhikevichNeuron
        n = SCIzhikevichNeuron(seed=42)
        assert n.v == n.c
        assert n.u == n.b * n.v
        state = n.get_state()
        assert "v" in state and "u" in state

    def test_step_no_spike(self):
        from sc_neurocore.neurons.sc_izhikevich import SCIzhikevichNeuron
        n = SCIzhikevichNeuron(seed=1)
        spike = n.step(0.0)
        assert spike == 0

    def test_step_spike(self):
        from sc_neurocore.neurons.sc_izhikevich import SCIzhikevichNeuron
        n = SCIzhikevichNeuron(seed=1)
        # Drive with strong current until spike
        spikes = [n.step(40.0) for _ in range(50)]
        assert 1 in spikes

    def test_noise_path(self):
        from sc_neurocore.neurons.sc_izhikevich import SCIzhikevichNeuron
        n = SCIzhikevichNeuron(noise_std=1.0, seed=42)
        n.step(5.0)  # covers noise branch

    def test_reset_state(self):
        from sc_neurocore.neurons.sc_izhikevich import SCIzhikevichNeuron
        n = SCIzhikevichNeuron(seed=1)
        for _ in range(10):
            n.step(30.0)
        n.reset_state()
        assert n.v == n.c

    def test_get_state(self):
        from sc_neurocore.neurons.sc_izhikevich import SCIzhikevichNeuron
        n = SCIzhikevichNeuron(seed=1)
        s = n.get_state()
        assert isinstance(s["v"], float)
        assert isinstance(s["u"], float)


class TestStochasticLIFCoverage:
    def test_refractory_path(self):
        from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron
        n = StochasticLIFNeuron(
            v_threshold=0.5, refractory_period=3, seed=42, noise_std=0.0
        )
        # Drive to spike
        for _ in range(50):
            if n.step(1.0):
                break
        # Now in refractory — step should return 0 and v == v_rest
        result = n.step(1.0)
        assert result == 0
        assert n.v == n.v_rest

    def test_entropy_source_path(self):
        from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron

        class FakeEntropy:
            def sample_normal(self, mean, std):
                return 0.01

        n = StochasticLIFNeuron(
            noise_std=0.1, entropy_source=FakeEntropy(), seed=42
        )
        n.step(0.5)  # covers entropy_source branch

    def test_get_state(self):
        from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron
        n = StochasticLIFNeuron(seed=1)
        s = n.get_state()
        assert "v" in s and "refractory" in s

    def test_process_bitstream(self):
        from sc_neurocore.neurons.stochastic_lif import StochasticLIFNeuron
        n = StochasticLIFNeuron(v_threshold=0.3, seed=42, noise_std=0.0)
        bits = np.ones(50, dtype=np.uint8)
        spikes = n.process_bitstream(bits, input_scale=0.5)
        assert spikes.shape == (50,)
        assert spikes.dtype == np.uint8


class TestHomeostaticLIFCoverage:
    def test_get_state(self):
        from sc_neurocore.neurons.homeostatic_lif import HomeostaticLIFNeuron
        n = HomeostaticLIFNeuron(seed=1)
        n.step(0.5)
        s = n.get_state()
        assert "threshold" in s and "rate_trace" in s


class TestDendriticNeuronCoverage:
    def test_step_xor_logic(self):
        from sc_neurocore.neurons.dendritic import StochasticDendriticNeuron
        n = StochasticDendriticNeuron()
        assert n.step(1.0, 0.0) == 1
        assert n.step(0.0, 1.0) == 1
        assert n.step(1.0, 1.0) == 0
        assert n.step(0.0, 0.0) == 0

    def test_reset_and_get_state(self):
        from sc_neurocore.neurons.dendritic import StochasticDendriticNeuron
        n = StochasticDendriticNeuron()
        n.step(1.0, 0.0)
        s = n.get_state()
        assert "last_current" in s
        n.reset_state()
        assert n._last_current == 0.0


class TestFixedPointLIFCoverage:
    def test_reset_state_alias(self):
        from sc_neurocore.neurons.fixed_point_lif import FixedPointLIFNeuron
        n = FixedPointLIFNeuron()
        n.step(26, 128, 200, 0)  # push v
        n.reset_state()
        assert n.v == n.v_rest

    def test_get_state(self):
        from sc_neurocore.neurons.fixed_point_lif import FixedPointLIFNeuron
        n = FixedPointLIFNeuron()
        s = n.get_state()
        assert "v" in s and "refractory_counter" in s

    def test_lfsr_reset_with_seed(self):
        from sc_neurocore.neurons.fixed_point_lif import FixedPointLFSR
        lfsr = FixedPointLFSR(seed=0xBEEF)
        lfsr.step()
        lfsr.reset(seed=0xCAFE)
        assert lfsr.reg == 0xCAFE

    def test_encoder_reset(self):
        from sc_neurocore.neurons.fixed_point_lif import FixedPointBitstreamEncoder
        enc = FixedPointBitstreamEncoder(seed_init=0xACE1)
        enc.step(128)
        enc.reset()


# ── core ─────────────────────────────────────────────────────────────
class TestTensorStreamCoverage:
    def test_to_bitstream_already_bitstream(self):
        from sc_neurocore.core.tensor_stream import TensorStream
        bits = np.random.randint(0, 2, (3, 10), dtype=np.uint8)
        ts = TensorStream(data=bits, domain="bitstream")
        result = ts.to_bitstream()
        np.testing.assert_array_equal(result, bits)

    def test_to_bitstream_invalid_domain(self):
        from sc_neurocore.core.tensor_stream import TensorStream
        ts = TensorStream(data=np.zeros(3), domain="unknown")
        with pytest.raises(ValueError):
            ts.to_bitstream()

    def test_to_prob_quantum_domain(self):
        from sc_neurocore.core.tensor_stream import TensorStream
        # Quantum state: (..., 2) complex
        qs = np.array([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]])
        ts = TensorStream(data=qs, domain="quantum")
        probs = ts.to_prob()
        np.testing.assert_allclose(probs, [0.0, 1.0])

    def test_to_prob_fallback(self):
        from sc_neurocore.core.tensor_stream import TensorStream
        ts = TensorStream(data=np.array([0.5, 0.6]), domain="spike")
        probs = ts.to_prob()
        np.testing.assert_allclose(probs, [0.5, 0.6])

    def test_to_quantum_already_quantum(self):
        from sc_neurocore.core.tensor_stream import TensorStream
        q = np.array([[1.0 + 0j, 0.0 + 0j]])
        ts = TensorStream(data=q, domain="quantum")
        result = ts.to_quantum()
        np.testing.assert_array_equal(result, q)


class TestReplication:
    def test_construction(self):
        from sc_neurocore.core.replication import VonNeumannProbe
        probe = VonNeumannProbe(probe_id=0)
        assert probe.probe_id == 0

    @pytest.mark.skipif(os.name == "nt", reason="copytree fails on Windows special files")
    def test_replicate(self):
        from sc_neurocore.core.replication import VonNeumannProbe
        probe = VonNeumannProbe(probe_id=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            dest = os.path.join(tmpdir, "replica")
            probe.replicate(dest)
            assert os.path.isdir(os.path.join(dest, "sc_neurocore"))
            assert os.path.isfile(os.path.join(dest, "launch_probe.py"))


class TestImmortalityCoverage:
    def test_capture_and_save_load(self):
        from sc_neurocore.core.immortality import DigitalSoul

        class FakeModule:
            def __init__(self):
                self.weights = np.array([1.0, 2.0])
            def get_state(self):
                return {"v": 0.5}

        class FakeOrchestrator:
            modules = {"layer1": FakeModule()}

        soul = DigitalSoul(agent_id="test")
        soul.capture_agent(FakeOrchestrator())
        assert "layer1_weights" in soul.state_data
        assert "layer1_state" in soul.state_data

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            soul.save_soul(path)
            loaded = DigitalSoul.load_soul(path)
            assert loaded.agent_id == "test"
        finally:
            os.unlink(path)

    def test_unpickler_blocks_unsafe(self):
        from sc_neurocore.core.immortality import _SoulUnpickler
        import pickle
        import io
        # Create a pickle that tries to import os.system
        malicious = pickle.dumps(os.system)
        with pytest.raises(pickle.UnpicklingError):
            _SoulUnpickler(io.BytesIO(malicious)).load()

    def test_reincarnate(self):
        from sc_neurocore.core.immortality import DigitalSoul

        class FakeModule:
            def __init__(self):
                self.weights = np.array([0.0, 0.0])
                self.v = 0.0
            def _refresh_packed_weights(self):
                pass

        class FakeOrchestrator:
            modules = {"layer1": FakeModule()}

        soul = DigitalSoul(agent_id="test", state_data={
            "layer1_weights": np.array([1.0, 2.0]),
            "layer1_state": {"v": 0.99}
        })
        orch = FakeOrchestrator()
        soul.reincarnate(orch)
        np.testing.assert_array_equal(orch.modules["layer1"].weights, [1.0, 2.0])
        assert orch.modules["layer1"].v == 0.99


class TestOrchestratorCoverage:
    def test_pipeline_with_forward_quantum(self):
        from sc_neurocore.core.orchestrator import CognitiveOrchestrator
        from sc_neurocore.core.tensor_stream import TensorStream

        class QuantumForwarder:
            __class__ = type("QuantumForwarder", (), {})
            def forward(self, x):
                return np.array([0.5 + 0.5j, 0.3 + 0.7j])

        # Monkey-patch name to include 'Quantum'
        QuantumForwarder.__name__ = "QuantumModule"

        orch = CognitiveOrchestrator()
        mod = QuantumForwarder()
        mod.__class__.__name__ = "QuantumModule"
        orch.register_module("qmod", mod)
        stream = TensorStream.from_prob(np.array([0.5, 0.6]))
        result = orch.execute_pipeline(["qmod"], stream)
        assert result.domain == "quantum"

    def test_pipeline_with_forward_uint8(self):
        from sc_neurocore.core.orchestrator import CognitiveOrchestrator
        from sc_neurocore.core.tensor_stream import TensorStream

        class BitModule:
            def forward(self, x):
                return np.array([0, 1, 1], dtype=np.uint8)

        orch = CognitiveOrchestrator()
        orch.register_module("bit", BitModule())
        stream = TensorStream.from_prob(np.array([0.5, 0.6, 0.7]))
        result = orch.execute_pipeline(["bit"], stream)
        assert result.domain == "bitstream"

    def test_pipeline_with_step_scalar(self):
        from sc_neurocore.core.orchestrator import CognitiveOrchestrator
        from sc_neurocore.core.tensor_stream import TensorStream

        class StepModule:
            def step(self, v):
                return float(v) * 2

        orch = CognitiveOrchestrator()
        orch.register_module("stepper", StepModule())
        stream = TensorStream.from_prob(np.float64(0.3))
        result = orch.execute_pipeline(["stepper"], stream)
        assert result.domain == "prob"

    def test_pipeline_missing_module(self):
        from sc_neurocore.core.orchestrator import CognitiveOrchestrator
        from sc_neurocore.core.tensor_stream import TensorStream
        orch = CognitiveOrchestrator()
        stream = TensorStream.from_prob(np.array([0.5]))
        result = orch.execute_pipeline(["nonexistent"], stream)
        assert result is stream


class TestMDLParserCoverage:
    def test_encode_module_with_weights_no_get_state(self):
        from sc_neurocore.core.mdl_parser import MindDescriptionLanguage

        class WeightModule:
            weights = np.array([1.0, 2.0, 3.0])

        class FakeOrch:
            modules = {"dense": WeightModule()}

        yaml_str = MindDescriptionLanguage.encode(FakeOrch(), "test_agent")
        assert "dense" in yaml_str
        data = MindDescriptionLanguage.decode(yaml_str)
        assert data["agent_name"] == "test_agent"


# ── utils/bitstreams ─────────────────────────────────────────────────
class TestBitstreamsCoverage:
    def test_generate_bernoulli_invalid_p(self):
        from sc_neurocore.utils.bitstreams import generate_bernoulli_bitstream
        with pytest.raises(ValueError):
            generate_bernoulli_bitstream(-0.1, 100)

    def test_generate_sobol(self):
        from sc_neurocore.utils.bitstreams import generate_sobol_bitstream
        bits = generate_sobol_bitstream(0.5, 64, seed=42)
        assert bits.shape == (64,)
        assert bits.dtype == np.uint8
        mean = bits.mean()
        assert 0.2 < mean < 0.8

    def test_generate_sobol_invalid_p(self):
        from sc_neurocore.utils.bitstreams import generate_sobol_bitstream
        with pytest.raises(ValueError):
            generate_sobol_bitstream(1.5, 100)

    def test_bitstream_to_probability_empty(self):
        from sc_neurocore.utils.bitstreams import bitstream_to_probability
        with pytest.raises(ValueError):
            bitstream_to_probability(np.array([], dtype=np.uint8))

    def test_value_to_unipolar_invalid_range(self):
        from sc_neurocore.utils.bitstreams import value_to_unipolar_prob
        with pytest.raises(ValueError):
            value_to_unipolar_prob(0.5, 1.0, 0.0)

    def test_unipolar_prob_to_value_invalid_p(self):
        from sc_neurocore.utils.bitstreams import unipolar_prob_to_value
        with pytest.raises(ValueError):
            unipolar_prob_to_value(1.5, 0.0, 1.0)

    def test_encoder_sobol_mode(self):
        from sc_neurocore.utils.bitstreams import BitstreamEncoder
        enc = BitstreamEncoder(x_min=0.0, x_max=1.0, length=64, mode="sobol", seed=1)
        bits = enc.encode(0.5)
        assert bits.shape == (64,)
        val = enc.decode(bits)
        assert 0.2 < val < 0.8

    def test_encoder_invalid_mode(self):
        from sc_neurocore.utils.bitstreams import BitstreamEncoder
        with pytest.raises(ValueError):
            BitstreamEncoder(x_min=0.0, x_max=1.0, mode="invalid")

    def test_averager_push_invalid_bit(self):
        from sc_neurocore.utils.bitstreams import BitstreamAverager
        avg = BitstreamAverager(window=10)
        with pytest.raises(ValueError):
            avg.push(5)

    def test_averager_wrapping(self):
        from sc_neurocore.utils.bitstreams import BitstreamAverager
        avg = BitstreamAverager(window=4)
        # Fill window: buffer = [0, 1, 0, 1], sum=2
        for b in [0, 1, 0, 1]:
            avg.push(b)
        assert avg._filled
        assert avg.estimate() == 0.5
        # Push 1 → replaces buffer[0]=0 with 1: buffer=[1,1,0,1], sum=3
        avg.push(1)
        assert avg.estimate() == 0.75

    def test_averager_empty_estimate(self):
        from sc_neurocore.utils.bitstreams import BitstreamAverager
        avg = BitstreamAverager(window=10)
        assert avg.estimate() == 0.0

    def test_averager_reset(self):
        from sc_neurocore.utils.bitstreams import BitstreamAverager
        avg = BitstreamAverager(window=4)
        avg.push(1)
        avg.push(1)
        avg.reset()
        assert avg.estimate() == 0.0


# ── utils/rng ────────────────────────────────────────────────────────
class TestRNGCoverage:
    def test_uniform(self):
        from sc_neurocore.utils.rng import RNG
        r = RNG(seed=42)
        val = r.uniform(0.0, 1.0)
        assert 0.0 <= val <= 1.0


# ── accel/vector_ops ─────────────────────────────────────────────────
class TestVectorOpsCoverage:
    def test_pack_unpack_2d(self):
        from sc_neurocore.accel.vector_ops import pack_bitstream, unpack_bitstream
        bits = np.random.randint(0, 2, (3, 128), dtype=np.uint8)
        packed = pack_bitstream(bits)
        assert packed.shape == (3, 2)
        unpacked = unpack_bitstream(packed, 3 * 128, original_shape=(3, 128))
        np.testing.assert_array_equal(unpacked, bits)

    def test_unpack_2d_no_shape(self):
        from sc_neurocore.accel.vector_ops import pack_bitstream, unpack_bitstream
        bits = np.random.randint(0, 2, (2, 64), dtype=np.uint8)
        packed = pack_bitstream(bits)
        unpacked = unpack_bitstream(packed, 128)
        np.testing.assert_array_equal(unpacked, bits)

    def test_pack_3d_raises(self):
        from sc_neurocore.accel.vector_ops import pack_bitstream
        with pytest.raises(ValueError):
            pack_bitstream(np.zeros((2, 2, 64), dtype=np.uint8))

    def test_unpack_3d_raises(self):
        from sc_neurocore.accel.vector_ops import unpack_bitstream
        with pytest.raises(ValueError):
            unpack_bitstream(np.zeros((2, 2, 2), dtype=np.uint64), 8)


# ── accel/gpu_backend ────────────────────────────────────────────────
class TestGPUBackendCoverage:
    """Test CPU fallback paths (GPU paths need CUDA)."""

    def test_to_device_cpu(self):
        from sc_neurocore.accel.gpu_backend import to_device, HAS_CUPY
        arr = np.array([1.0, 2.0])
        result = to_device(arr)
        if not HAS_CUPY:
            assert result is arr

    def test_to_host_cpu(self):
        from sc_neurocore.accel.gpu_backend import to_host
        arr = np.array([1.0, 2.0])
        result = to_host(arr)
        np.testing.assert_array_equal(result, arr)

    def test_gpu_pack_1d(self):
        from sc_neurocore.accel.gpu_backend import gpu_pack_bitstream
        bits = np.random.randint(0, 2, 128, dtype=np.uint8)
        packed = gpu_pack_bitstream(bits)
        assert packed.shape == (2,)

    def test_gpu_pack_2d(self):
        from sc_neurocore.accel.gpu_backend import gpu_pack_bitstream
        bits = np.random.randint(0, 2, (3, 64), dtype=np.uint8)
        packed = gpu_pack_bitstream(bits)
        assert packed.shape == (3, 1)

    def test_gpu_pack_3d_raises(self):
        from sc_neurocore.accel.gpu_backend import gpu_pack_bitstream
        with pytest.raises(ValueError):
            gpu_pack_bitstream(np.zeros((2, 2, 64), dtype=np.uint8))

    def test_gpu_vec_and(self):
        from sc_neurocore.accel.gpu_backend import gpu_vec_and
        a = np.array([0xFF, 0x00], dtype=np.uint64)
        b = np.array([0x0F, 0xFF], dtype=np.uint64)
        result = gpu_vec_and(a, b)
        np.testing.assert_array_equal(result, [0x0F, 0x00])

    def test_gpu_popcount(self):
        from sc_neurocore.accel.gpu_backend import gpu_popcount
        packed = np.array([0xFFFFFFFFFFFFFFFF], dtype=np.uint64)
        count = gpu_popcount(packed)
        assert int(count[0]) == 64

    def test_gpu_vec_mac(self):
        from sc_neurocore.accel.gpu_backend import gpu_vec_mac
        # Simple: 1 neuron, 1 input, 1 word — all ones
        w = np.array([[[0xFFFFFFFFFFFFFFFF]]], dtype=np.uint64)  # (1,1,1)
        inp = np.array([[0xFFFFFFFFFFFFFFFF]], dtype=np.uint64)  # (1,1)
        result = gpu_vec_mac(w, inp)
        assert result[0] == 64


# ── synapses ─────────────────────────────────────────────────────────
class TestSynapseCoverage:
    def test_synapse_invalid_range(self):
        from sc_neurocore.synapses.sc_synapse import BitstreamSynapse
        with pytest.raises(ValueError):
            BitstreamSynapse(w_min=1.0, w_max=0.0)

    def test_synapse_length_mismatch(self):
        from sc_neurocore.synapses.sc_synapse import BitstreamSynapse
        syn = BitstreamSynapse(w_min=0.0, w_max=1.0, length=10)
        with pytest.raises(ValueError):
            syn.apply(np.ones(20, dtype=np.uint8))


class TestDotProductCoverage:
    def test_empty_synapses(self):
        from sc_neurocore.synapses.dot_product import BitstreamDotProduct
        with pytest.raises(ValueError):
            BitstreamDotProduct(synapses=[])

    def test_input_count_mismatch(self):
        from sc_neurocore.synapses.dot_product import BitstreamDotProduct
        from sc_neurocore.synapses.sc_synapse import BitstreamSynapse
        syn = BitstreamSynapse(w_min=0.0, w_max=1.0, length=10)
        dp = BitstreamDotProduct(synapses=[syn])
        with pytest.raises(ValueError):
            dp.apply(np.ones((2, 10), dtype=np.uint8))


# ── layers ───────────────────────────────────────────────────────────
class TestSCDenseLayerCoverage:
    def test_neuron_params_none(self):
        from sc_neurocore.layers.sc_dense_layer import SCDenseLayer
        layer = SCDenseLayer(
            n_neurons=2, x_inputs=[0.5], weight_values=[0.5],
            x_min=0.0, x_max=1.0, w_min=0.0, w_max=1.0,
            length=64, base_seed=42, neuron_params=None,
        )
        # neuron_params should be defaulted to {}
        assert layer.neuron_params == {}


# ── model_bridge ─────────────────────────────────────────────────────
class TestModelBridgeCoverage:
    def test_load_shape_mismatch(self):
        from sc_neurocore.utils.model_bridge import SCBridge

        class FakeLayer:
            weights = np.zeros((2, 2))

        state = {"dense.weight": np.array([1.0, 2.0, 3.0])}  # shape (3,) != (2,2)
        SCBridge.load_from_state_dict(state, {"dense": FakeLayer()})

    def test_load_no_weights_attr(self):
        from sc_neurocore.utils.model_bridge import SCBridge

        class FakeLayer:
            pass

        state = {"dense.weight": np.array([1.0, 2.0])}
        SCBridge.load_from_state_dict(state, {"dense": FakeLayer()})

    def test_load_no_key(self):
        from sc_neurocore.utils.model_bridge import SCBridge

        class FakeLayer:
            weights = np.zeros(2)

        SCBridge.load_from_state_dict({}, {"dense": FakeLayer()})

    def test_export(self):
        from sc_neurocore.utils.model_bridge import SCBridge

        class FakeLayer:
            weights = np.array([0.5, 0.6])

        result = SCBridge.export_to_numpy({"dense": FakeLayer()})
        assert "dense.weight" in result
