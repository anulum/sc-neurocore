"""Tests for 0%-coverage modules: interfaces, generative, verification, viz,
world_model, pipeline, models, math, ensembles, analysis/qualia, accel."""

import numpy as np
import pytest


# ── interfaces ───────────────────────────────────────────────────────
class TestInterstellarDTN:
    def test_receive_and_step(self):
        from sc_neurocore.interfaces.interstellar import InterstellarDTN, Packet
        dtn = InterstellarDTN(node_id="node_0", link_availability=0.9)
        pkt = Packet(id=1, data=np.random.randn(4))
        dtn.receive(pkt)
        result = dtn.step()
        # result is Optional[Packet] — may be None if link unavailable
        assert result is None or hasattr(result, "data")


class TestPlanetarySensorGrid:
    def test_aggregate_field(self):
        from sc_neurocore.interfaces.planetary import PlanetarySensorGrid
        p = PlanetarySensorGrid(n_nodes=100)
        telemetry = {"temperature": np.random.randn(100), "pressure": np.random.randn(100)}
        result = p.aggregate_field(telemetry)
        assert isinstance(result, np.ndarray)


class TestRealWorldBridges:
    def test_lsl_bridge_construction(self):
        from sc_neurocore.interfaces.real_world import LSLBridge
        b = LSLBridge(stream_name="TestStream")
        chunk = b.receive_chunk(max_samples=4)
        assert isinstance(chunk, np.ndarray)

    def test_ros2_node_construction(self):
        from sc_neurocore.interfaces.real_world import ROS2Node
        n = ROS2Node(node_name="test_node")
        result = n.publish_cmd_vel(linear_x=0.5, angular_z=0.1)
        assert isinstance(result, bool)


class TestSymbiosisProtocol:
    def test_encode_decode(self):
        from sc_neurocore.interfaces.symbiosis import SymbiosisProtocol
        s = SymbiosisProtocol()
        vec = np.random.randn(8)
        encoded = s.encode_thought(vec, urgency=0.8)
        assert isinstance(encoded, np.ndarray)
        decoded = s.decode_sensation(encoded)
        assert isinstance(decoded, str)


# ── generative ───────────────────────────────────────────────────────
class TestSCAudioSynthesizer:
    def test_synthesize_tone(self):
        from sc_neurocore.generative.audio_synthesis import SCAudioSynthesizer
        g = SCAudioSynthesizer(sample_rate=1000)
        wave = g.synthesize_tone(frequency=440.0, duration_ms=10, probability=0.7)
        assert isinstance(wave, np.ndarray)
        assert wave.shape[0] > 0

    def test_bitstream_to_audio(self):
        from sc_neurocore.generative.audio_synthesis import SCAudioSynthesizer
        g = SCAudioSynthesizer(sample_rate=1000)
        bs = np.random.randint(0, 2, 64).astype(np.uint8)
        audio = g.bitstream_to_audio(bs)
        assert isinstance(audio, np.ndarray)


class TestSCTextGenerator:
    def test_generate_token(self):
        from sc_neurocore.generative.text_gen import SCTextGenerator
        vocab = ["hello", "world", "foo", "bar"]
        g = SCTextGenerator(vocab=vocab)
        prob = np.array([0.1, 0.5, 0.2, 0.2])
        token = g.generate_token(prob)
        assert token in vocab

    def test_generate_sequence(self):
        from sc_neurocore.generative.text_gen import SCTextGenerator
        vocab = ["a", "b", "c"]
        g = SCTextGenerator(vocab=vocab)
        seq = g.generate_sequence(length=5)
        assert isinstance(seq, str)


class TestSC3DGenerator:
    def test_bitstream_to_voxels(self):
        from sc_neurocore.generative.three_d_gen import SC3DGenerator
        g = SC3DGenerator(iso_level=0.5)
        bs = np.random.randint(0, 2, (4, 64)).astype(np.uint8)
        voxels = g.bitstream_to_voxels(bs, grid_size=(4, 4, 4))
        assert isinstance(voxels, np.ndarray)

    def test_generate_surface_mesh(self):
        from sc_neurocore.generative.three_d_gen import SC3DGenerator
        g = SC3DGenerator(iso_level=0.3)
        # Create a simple voxel grid with a spherical shape
        grid = np.zeros((8, 8, 8))
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    grid[i, j, k] = 1.0 if (i-3.5)**2 + (j-3.5)**2 + (k-3.5)**2 < 6.0 else 0.0
        mesh = g.generate_surface_mesh(grid)
        assert isinstance(mesh, dict)


# ── verification ─────────────────────────────────────────────────────
class TestFormalVerifier:
    def test_verify_probability_bounds(self):
        from sc_neurocore.verification.formal_proofs import FormalVerifier, Interval
        inp = Interval(min_val=0.0, max_val=1.0)
        wt = Interval(min_val=0.0, max_val=1.0)
        result = FormalVerifier.verify_probability_bounds(inp, wt)
        assert isinstance(result, bool)

    def test_verify_energy_safety(self):
        from sc_neurocore.verification.formal_proofs import FormalVerifier
        result = FormalVerifier.verify_energy_safety(energy=0.5, cost=0.3)
        assert isinstance(result, bool)

    def test_interval_ops(self):
        from sc_neurocore.verification.formal_proofs import Interval
        a = Interval(0.2, 0.8)
        b = Interval(0.1, 0.5)
        c = a + b
        assert c.min_val == pytest.approx(0.3, abs=0.01)
        d = a * b
        assert isinstance(d, Interval)


class TestCodeSafetyVerifier:
    def test_verify_code_safety(self):
        from sc_neurocore.verification.safety import CodeSafetyVerifier
        v = CodeSafetyVerifier()
        safe = v.verify_code_safety("x = 1 + 2")
        assert isinstance(safe, bool)

    def test_verify_logic_invariant(self):
        from sc_neurocore.verification.safety import CodeSafetyVerifier
        v = CodeSafetyVerifier()
        result = v.verify_logic_invariant(
            func=lambda x: x * 2,
            input_sample=3,
            expected_condition=lambda out: out == 6,
        )
        assert result is True


# ── viz ──────────────────────────────────────────────────────────────
class TestNeuroArtGenerator:
    def test_generate_visual(self):
        from sc_neurocore.viz.neuro_art import NeuroArtGenerator
        g = NeuroArtGenerator(resolution=16)
        state = np.random.randn(8)
        img = g.generate_visual(state)
        assert isinstance(img, np.ndarray)


class TestWebVisualizer:
    def test_generate_html(self):
        from sc_neurocore.viz.web_viz import WebVisualizer
        html = WebVisualizer.generate_html(layers=[{"name": "L1", "n": 4}])
        # generate_html writes a file or returns a string
        assert html is None or isinstance(html, str)


# ── world_model ──────────────────────────────────────────────────────
class TestPredictiveWorldModel:
    def test_predict_next_state(self):
        from sc_neurocore.world_model.predictive_model import PredictiveWorldModel
        m = PredictiveWorldModel(state_dim=4, action_dim=2)
        pred = m.predict_next_state(
            current_state=np.zeros(4),
            action=np.zeros(2),
        )
        assert pred.shape[0] == 4

    def test_forecast(self):
        from sc_neurocore.world_model.predictive_model import PredictiveWorldModel
        m = PredictiveWorldModel(state_dim=4, action_dim=2)
        actions = [np.zeros(2), np.ones(2)]
        trajectory = m.forecast(initial_state=np.zeros(4), actions=actions)
        assert len(trajectory) == 2


class TestSCPlanner:
    def test_propose_action(self):
        from sc_neurocore.world_model.predictive_model import PredictiveWorldModel
        from sc_neurocore.world_model.planner import SCPlanner
        wm = PredictiveWorldModel(state_dim=4, action_dim=2)
        p = SCPlanner(world_model=wm)
        action = p.propose_action(
            current_state=np.zeros(4),
            goal_state=np.ones(4),
            n_candidates=5,
        )
        assert isinstance(action, np.ndarray)

    def test_plan_sequence(self):
        from sc_neurocore.world_model.predictive_model import PredictiveWorldModel
        from sc_neurocore.world_model.planner import SCPlanner
        wm = PredictiveWorldModel(state_dim=4, action_dim=2)
        p = SCPlanner(world_model=wm)
        plan = p.plan_sequence(
            current_state=np.zeros(4),
            goal_state=np.ones(4),
            horizon=3,
        )
        assert len(plan) > 0


# ── pipeline ─────────────────────────────────────────────────────────
class TestDataIngestor:
    def test_prepare_dataset(self):
        from sc_neurocore.pipeline.ingestion import DataIngestor
        d = DataIngestor()
        raw = {"signals": np.random.randn(10, 4), "labels": np.array([0, 1] * 5)}
        ds = d.prepare_dataset(raw)
        assert ds is not None

    def test_multimodal_dataset_get_sample(self):
        from sc_neurocore.pipeline.ingestion import MultimodalDataset
        data = {"eeg": np.random.randn(10, 4), "emg": np.random.randn(10, 2)}
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        ds = MultimodalDataset(data=data, labels=labels)
        sample = ds.get_sample(0)
        assert isinstance(sample, dict)


class TestSCTrainingLoop:
    def test_train_multimodal_fusion(self):
        from sc_neurocore.pipeline.training import SCTrainingLoop
        from sc_neurocore.pipeline.ingestion import MultimodalDataset
        data = {"signals": np.random.randn(5, 4)}
        labels = np.array([0, 1, 0, 1, 0])
        ds = MultimodalDataset(data=data, labels=labels)
        # Need a fusion layer — use a simple mock
        class MockFusion:
            def train_step(self, batch):
                pass
        SCTrainingLoop.train_multimodal_fusion(MockFusion(), ds, epochs=1)


# ── models ───────────────────────────────────────────────────────────
class TestSCDigitClassifier:
    def test_forward(self):
        from sc_neurocore.models.zoo import SCDigitClassifier
        clf = SCDigitClassifier()
        img = np.random.rand(28, 28)  # 28x28 image expected
        digit = clf.forward(img)
        assert isinstance(digit, (int, np.integer))


class TestSCKeywordSpotter:
    def test_predict(self):
        from sc_neurocore.models.zoo import SCKeywordSpotter
        s = SCKeywordSpotter(n_keywords=3)
        features = np.random.rand(16)  # expects 16 features (n_inputs=16)
        kw = s.predict(features)
        assert isinstance(kw, (int, np.integer))


# ── math ─────────────────────────────────────────────────────────────
class TestCategoryTheoryBridge:
    def test_stochastic_to_quantum(self):
        from sc_neurocore.math.category_theory import CategoryTheoryBridge
        bs = np.random.randint(0, 2, 64).astype(np.uint8)
        result = CategoryTheoryBridge.stochastic_to_quantum(bs)
        assert isinstance(result, np.ndarray)

    def test_quantum_to_bio(self):
        from sc_neurocore.math.category_theory import CategoryTheoryBridge
        sv = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
        result = CategoryTheoryBridge.quantum_to_bio(sv)
        assert isinstance(result, float)

    def test_bio_to_stochastic(self):
        from sc_neurocore.math.category_theory import CategoryTheoryBridge
        bs = CategoryTheoryBridge.bio_to_stochastic(0.7, length=32)
        assert isinstance(bs, np.ndarray)
        assert bs.shape == (32,)

    def test_morphism_and_category_object(self):
        from sc_neurocore.math.category_theory import CategoryObject, Morphism
        obj = CategoryObject(data=np.array([1.0, 2.0]), domain="stochastic")
        # Morphism.__call__ passes obj.data to func, then wraps result
        m = Morphism(func=lambda x: x * 2, name="double")
        result = m(obj)
        assert result.data[0] == pytest.approx(2.0)
        assert result.domain == "double"

    def test_get_functor(self):
        from sc_neurocore.math.category_theory import CategoryTheoryBridge
        bridge = CategoryTheoryBridge()
        # Domain names are capitalized
        functor = bridge.get_functor("Stochastic", "Quantum")
        assert functor is not None


# ── ensembles ────────────────────────────────────────────────────────
class TestEnsembleOrchestrator:
    def test_construction_and_add(self):
        from sc_neurocore.ensembles.orchestrator import EnsembleOrchestrator
        e = EnsembleOrchestrator()
        assert isinstance(e.agents, dict)
        assert len(e.agents) == 0

    def test_coordinated_mission(self):
        from sc_neurocore.ensembles.orchestrator import EnsembleOrchestrator
        e = EnsembleOrchestrator()
        # coordinated_mission just prints, no agents needed
        e.coordinated_mission("test_goal")


# ── analysis/qualia ──────────────────────────────────────────────────
class TestQualiaTuringTest:
    def test_administer(self):
        from sc_neurocore.transcendent.noetic import SemioticTriad
        from sc_neurocore.analysis.qualia import QualiaTuringTest
        sem = SemioticTriad()
        sem.learn_association("Fire", "Heat")
        sem.learn_association("Ocean", "Calm")
        sem.learn_association("Void", "Emptiness")
        q = QualiaTuringTest(semiotics=sem)
        state = np.array([0.9, 0.1, 0.05])
        result = q.administer_test(state)
        assert isinstance(result, bool)


# ── accel/jit_kernels ────────────────────────────────────────────────
class TestJITKernels:
    def test_jit_pack_bits(self):
        from sc_neurocore.accel.jit_kernels import jit_pack_bits
        bitstream = np.array([1, 0, 1, 0, 1, 0, 1, 0] * 8, dtype=np.uint8)
        packed = np.zeros(1, dtype=np.uint64)
        jit_pack_bits(bitstream, packed)
        assert packed[0] != 0

    def test_jit_vec_mac(self):
        from sc_neurocore.accel.jit_kernels import jit_vec_mac
        # weights shape: (n_neurons, n_inputs, n_words) = (2, 3, 1)
        pw = np.array([[[0xFF], [0x0F], [0xF0]],
                        [[0xAA], [0x55], [0xFF]]], dtype=np.uint64)
        # inputs shape: (n_inputs, n_words) = (3, 1)
        pi = np.array([[0x0F], [0xFF], [0x0F]], dtype=np.uint64)
        out = np.zeros(2, dtype=np.float64)
        jit_vec_mac(pw, pi, out)
        assert out[0] > 0


# ── accel/mpi_driver ─────────────────────────────────────────────────
class TestMPIDriver:
    def test_construction(self):
        from sc_neurocore.accel.mpi_driver import MPIDriver
        d = MPIDriver()
        assert d is not None

    def test_scatter_gather(self):
        from sc_neurocore.accel.mpi_driver import MPIDriver
        d = MPIDriver()
        data = np.random.randn(8)
        local = d.scatter_workload(data)
        assert isinstance(local, np.ndarray)
        result = d.gather_results(local)
        assert isinstance(result, np.ndarray)

    def test_barrier(self):
        from sc_neurocore.accel.mpi_driver import MPIDriver
        d = MPIDriver()
        d.barrier()


# ── SCPN layers ──────────────────────────────────────────────────────
class TestSCPNLayers:
    def test_l1_quantum_layer(self):
        from sc_neurocore.scpn.layers.l1_quantum import L1_QuantumLayer, L1_StochasticParameters
        params = L1_StochasticParameters(n_qubits=8, bitstream_length=32)
        layer = L1_QuantumLayer(params)
        output = layer.step(0.01)
        assert output.shape == (8, 32)
        metric = layer.get_global_metric()
        assert 0.0 <= metric <= 1.0

    def test_l1_with_external_field(self):
        from sc_neurocore.scpn.layers.l1_quantum import L1_QuantumLayer, L1_StochasticParameters
        params = L1_StochasticParameters(n_qubits=8, bitstream_length=32)
        layer = L1_QuantumLayer(params)
        field = np.random.uniform(0, 1, 8)
        output = layer.step(0.01, external_field=field)
        assert output.shape == (8, 32)

    def test_l2_neurochemical(self):
        from sc_neurocore.scpn.layers.l2_neurochemical import L2_NeurochemicalLayer, L2_StochasticParameters
        params = L2_StochasticParameters(n_receptors=8, bitstream_length=32)
        layer = L2_NeurochemicalLayer(params)
        output = layer.step(0.01)
        assert output is not None
        assert layer.get_global_metric() >= 0.0

    def test_l3_genomic(self):
        from sc_neurocore.scpn.layers.l3_genomic import L3_GenomicLayer, L3_StochasticParameters
        params = L3_StochasticParameters(n_genes=8, bitstream_length=32)
        layer = L3_GenomicLayer(params)
        output = layer.step(0.01)
        assert output is not None

    def test_l4_cellular(self):
        from sc_neurocore.scpn.layers.l4_cellular import L4_CellularLayer, L4_StochasticParameters
        params = L4_StochasticParameters(grid_size=(4, 4), bitstream_length=32)
        layer = L4_CellularLayer(params)
        output = layer.step(0.01)
        assert output is not None

    def test_l5_organismal(self):
        from sc_neurocore.scpn.layers.l5_organismal import L5_OrganismalLayer, L5_StochasticParameters
        params = L5_StochasticParameters(n_emotional_dims=8, n_autonomic_nodes=16, bitstream_length=32)
        layer = L5_OrganismalLayer(params)
        output = layer.step(0.01)
        assert output is not None

    def test_l6_ecological(self):
        from sc_neurocore.scpn.layers.l6_ecological import L6_EcologicalLayer, L6_StochasticParameters
        params = L6_StochasticParameters(n_field_nodes=16, bitstream_length=32)
        layer = L6_EcologicalLayer(params)
        output = layer.step(0.01)
        assert output is not None

    def test_l7_symbolic(self):
        from sc_neurocore.scpn.layers.l7_symbolic import L7_SymbolicLayer, L7_StochasticParameters
        params = L7_StochasticParameters(n_symbols=8, bitstream_length=32)
        layer = L7_SymbolicLayer(params)
        output = layer.step(0.01)
        assert output is not None

    def test_create_full_stack(self):
        from sc_neurocore.scpn.layers import create_full_stack, get_global_metrics
        stack = create_full_stack()
        assert len(stack) == 7
        metrics = get_global_metrics(stack)
        assert len(metrics) == 7

    def test_run_integrated_step(self):
        from sc_neurocore.scpn.layers import create_full_stack, run_integrated_step
        stack = create_full_stack()
        outputs = run_integrated_step(stack, dt=0.01)
        assert "l1" in outputs and "l7" in outputs
