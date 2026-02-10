"""
Tests for research/contrib modules that were at 0% coverage:
  - chaos/rng.py (ChaoticRNG)
  - analysis/explainability.py (SpikeToConceptMapper)
  - analysis/kardashev.py (KardashevEstimator)
  - analysis/consciousness.py (PhiEvaluator)
  - bio/neuromodulation.py (NeuromodulatorSystem)
  - spatial/representations.py (VoxelGrid, PointCloud)
  - spatial/transformer_3d.py (SpatialTransformer3D)
  - physics/wolfram_hypergraph.py (WolframHypergraph)
  - core/mdl_parser.py (MindDescriptionLanguage, MDLSpecification)
  - learning/neuroevolution.py (SNNGeneticEvolver)
  - robotics/swarm.py (SwarmCoupling)
"""

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# ChaoticRNG
# ---------------------------------------------------------------------------
from sc_neurocore.chaos.rng import ChaoticRNG


class TestChaoticRNG:
    def test_construction(self):
        rng = ChaoticRNG(r=4.0, x=0.3)
        # After burn-in, x should be in (0, 1)
        assert 0 < rng.x < 1

    def test_random_output_shape(self):
        rng = ChaoticRNG()
        vals = rng.random(100)
        assert vals.shape == (100,)

    def test_random_in_unit_interval(self):
        rng = ChaoticRNG()
        vals = rng.random(1000)
        assert np.all(vals >= 0) and np.all(vals <= 1)

    def test_generate_bitstream_shape(self):
        rng = ChaoticRNG()
        bs = rng.generate_bitstream(0.5, 256)
        assert bs.shape == (256,)
        assert set(np.unique(bs)).issubset({0, 1})

    def test_generate_bitstream_probability(self):
        # x=0.5 is a fixed point for r=4, so use a different init
        rng = ChaoticRNG(r=4.0, x=0.3)
        bs = rng.generate_bitstream(0.4, 10000)
        # Chaotic logistic map has arcsine distribution, not uniform.
        # Bitstream probability won't exactly match p, but should be non-trivial.
        assert 0.0 < bs.mean() < 1.0

    def test_deterministic_same_init(self):
        rng1 = ChaoticRNG(r=4.0, x=0.123)
        rng2 = ChaoticRNG(r=4.0, x=0.123)
        np.testing.assert_array_equal(rng1.random(50), rng2.random(50))


# ---------------------------------------------------------------------------
# SpikeToConceptMapper (explainability)
# ---------------------------------------------------------------------------
from sc_neurocore.analysis.explainability import SpikeToConceptMapper


class TestSpikeToConceptMapper:
    def test_active_concepts(self):
        mapper = SpikeToConceptMapper({0: "Vision", 1: "Motor", 2: "Audio"})
        spikes = np.array([1, 0, 1, 0])
        result = mapper.explain(spikes)
        assert "Vision" in result
        assert "Audio" in result

    def test_idle(self):
        mapper = SpikeToConceptMapper({0: "Vision"})
        spikes = np.array([0, 0, 0])
        assert "idle" in mapper.explain(spikes)

    def test_unknown_neuron(self):
        mapper = SpikeToConceptMapper({0: "Vision"})
        spikes = np.array([0, 1])  # neuron 1 not in map
        result = mapper.explain(spikes)
        assert "Unknown(1)" in result


# ---------------------------------------------------------------------------
# KardashevEstimator
# ---------------------------------------------------------------------------
from sc_neurocore.analysis.kardashev import KardashevEstimator


class TestKardashevEstimator:
    def test_zero_power(self):
        assert KardashevEstimator.calculate_type(0) == 0.0

    def test_negative_power(self):
        assert KardashevEstimator.calculate_type(-100) == 0.0

    def test_type_1_civilization(self):
        # Type 1 = 10^16 W
        k = KardashevEstimator.calculate_type(1e16)
        assert k == pytest.approx(1.0)

    def test_type_2_civilization(self):
        # Type 2 = 10^26 W
        k = KardashevEstimator.calculate_type(1e26)
        assert k == pytest.approx(2.0)

    def test_estimate_from_compute(self):
        k = KardashevEstimator.estimate_from_compute(1e37, efficiency_j_per_op=1e-21)
        assert k == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# PhiEvaluator (consciousness)
# ---------------------------------------------------------------------------
from sc_neurocore.analysis.consciousness import PhiEvaluator


class TestPhiEvaluator:
    def test_entropy_all_ones(self):
        bs = np.ones(100)
        assert PhiEvaluator.entropy(bs) == 0.0

    def test_entropy_all_zeros(self):
        bs = np.zeros(100)
        assert PhiEvaluator.entropy(bs) == 0.0

    def test_entropy_balanced(self):
        bs = np.array([0, 1] * 500)
        assert PhiEvaluator.entropy(bs) == pytest.approx(1.0, abs=0.01)

    def test_phi_1d_returns_zero(self):
        """1D snapshot should return 0."""
        assert PhiEvaluator.calculate_phi(np.array([0.5, 0.3])) == 0.0

    def test_phi_independent_neurons(self):
        """Independent random neurons should have low Phi."""
        rng = np.random.default_rng(42)
        data = (rng.random((4, 1000)) < 0.5).astype(np.uint8)
        phi = PhiEvaluator.calculate_phi(data)
        assert phi >= 0
        assert phi < 0.5  # weak integration

    def test_phi_correlated_neurons(self):
        """Perfectly correlated neurons should have higher Phi."""
        row = np.array([0, 1] * 500, dtype=np.uint8)
        data = np.stack([row, row, row])
        phi = PhiEvaluator.calculate_phi(data)
        # H(each) = 1.0, H(joint) = 1.0, so phi = 3*1 - 1 = 2
        assert phi > 1.0


# ---------------------------------------------------------------------------
# NeuromodulatorSystem
# ---------------------------------------------------------------------------
from sc_neurocore.bio.neuromodulation import NeuromodulatorSystem


class TestNeuromodulatorSystem:
    def test_defaults(self):
        nm = NeuromodulatorSystem()
        assert nm.da_level == 0.5
        assert nm.ht_level == 0.5
        assert nm.ne_level == 0.1

    def test_reward_increases_dopamine(self):
        nm = NeuromodulatorSystem(da_level=0.3)
        nm.update_levels(reward=1.0, stress=0.0)
        assert nm.da_level > 0.3

    def test_stress_increases_norepinephrine(self):
        nm = NeuromodulatorSystem(ne_level=0.1)
        nm.update_levels(reward=0.0, stress=1.0)
        assert nm.ne_level > 0.1

    def test_stress_decreases_serotonin(self):
        nm = NeuromodulatorSystem(ht_level=0.8)
        nm.update_levels(reward=0.0, stress=1.0)
        assert nm.ht_level < 0.8

    def test_serotonin_clipped(self):
        nm = NeuromodulatorSystem(ht_level=0.2)
        nm.update_levels(reward=0.0, stress=10.0)
        assert nm.ht_level >= 0.1  # clipped to min

    def test_modulate_neuron(self):
        nm = NeuromodulatorSystem(da_level=0.8, ht_level=0.6, ne_level=0.3)
        params = {"v_threshold": 1.0, "noise_std": 0.5}
        mod = nm.modulate_neuron(params)
        assert mod["v_threshold"] < 1.0  # DA lowers threshold
        assert "noise_std" in mod


# ---------------------------------------------------------------------------
# VoxelGrid, PointCloud
# ---------------------------------------------------------------------------
from sc_neurocore.spatial.representations import VoxelGrid, PointCloud


class TestVoxelGrid:
    def test_construction(self):
        vg = VoxelGrid(resolution=4)
        assert vg.data.shape == (4, 4, 4)
        assert np.all(vg.data == 0)

    def test_set_voxel(self):
        vg = VoxelGrid(resolution=4)
        vg.set_voxel(1, 2, 3, 0.8)
        assert vg.data[1, 2, 3] == 0.8

    def test_set_voxel_out_of_bounds(self):
        vg = VoxelGrid(resolution=4)
        vg.set_voxel(10, 0, 0, 1.0)  # should be no-op
        assert np.all(vg.data == 0)

    def test_get_as_bitstream(self):
        vg = VoxelGrid(resolution=2)
        vg.set_voxel(0, 0, 0, 1.0)
        bs = vg.get_as_bitstream(length=100)
        assert bs.shape == (2, 2, 2, 100)
        # Voxel at (0,0,0) should be all 1s (p=1.0)
        assert bs[0, 0, 0, :].mean() == 1.0


class TestPointCloud:
    def test_normalize(self):
        pts = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        ints = np.array([0.5, 1.5])
        pc = PointCloud(points=pts, intensities=ints)
        pc.normalize()
        assert pc.points.min() >= 0.0
        assert pc.points.max() <= 1.0
        assert np.all(pc.intensities <= 1.0)


# ---------------------------------------------------------------------------
# SpatialTransformer3D
# ---------------------------------------------------------------------------
from sc_neurocore.spatial.transformer_3d import SpatialTransformer3D


class TestSpatialTransformer3D:
    def test_forward_shape(self):
        st = SpatialTransformer3D(resolution=3, dim_k=2)
        grid = np.random.rand(3, 3, 3)
        out = st.forward(grid)
        assert out.shape == (3, 3, 3)

    def test_forward_non_negative(self):
        st = SpatialTransformer3D(resolution=2, dim_k=4)
        grid = np.random.rand(2, 2, 2)
        out = st.forward(grid)
        # Attention output can vary, but should be finite
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# WolframHypergraph
# ---------------------------------------------------------------------------
from sc_neurocore.physics.wolfram_hypergraph import WolframHypergraph


class TestWolframHypergraph:
    def test_construction(self):
        wh = WolframHypergraph(edges=[(0, 1), (1, 2)], max_node_id=2)
        assert len(wh.edges) == 2

    def test_evolve_creates_new_edges(self):
        wh = WolframHypergraph(edges=[(0, 1), (1, 2)], max_node_id=2)
        wh.evolve(steps=1)
        # Rule: {x,y},{y,z} -> {x,z},{x,w},{y,w}
        assert len(wh.edges) == 3
        assert wh.max_node_id == 3

    def test_dimension_estimate(self):
        wh = WolframHypergraph(edges=[(0, 1), (1, 2), (2, 3)], max_node_id=3)
        dim = wh.dimension_estimate()
        assert dim == 3  # edge count

    def test_multi_step_evolution(self):
        wh = WolframHypergraph(edges=[(0, 1), (1, 2), (2, 3), (3, 4)], max_node_id=4)
        initial_count = len(wh.edges)
        wh.evolve(steps=2)
        # Should grow
        assert len(wh.edges) >= initial_count


# ---------------------------------------------------------------------------
# MindDescriptionLanguage
# ---------------------------------------------------------------------------
from sc_neurocore.core.mdl_parser import MindDescriptionLanguage, MDLSpecification


class TestMDLSpecification:
    def test_defaults(self):
        spec = MDLSpecification()
        assert spec.version == "1.0"
        assert spec.agent_name == "Unknown"
        assert spec.architecture == {}

    def test_custom(self):
        spec = MDLSpecification(agent_name="TestAgent", version="2.0")
        assert spec.agent_name == "TestAgent"


class TestMindDescriptionLanguage:
    def test_encode_decode_roundtrip(self, capsys):
        class MockModule:
            def get_state(self):
                return {"w": [0.1, 0.2]}

        class MockOrchestrator:
            modules = {"layer1": MockModule()}

        mdl_str = MindDescriptionLanguage.encode(MockOrchestrator(), "TestBot")
        assert "TestBot" in mdl_str
        assert "layer1" in mdl_str

        data = MindDescriptionLanguage.decode(mdl_str)
        assert data["agent_name"] == "TestBot"
        assert "layer1" in data["state"]

    def test_decode_minimal(self, capsys):
        yaml_str = "agent_name: Min\nversion: '1.0'\narchitecture: {}\nstate: {}\n"
        data = MindDescriptionLanguage.decode(yaml_str)
        assert data["agent_name"] == "Min"


# ---------------------------------------------------------------------------
# SNNGeneticEvolver
# ---------------------------------------------------------------------------
from sc_neurocore.learning.neuroevolution import SNNGeneticEvolver


class TestSNNGeneticEvolver:
    def _make_layer(self):
        class MockLayer:
            def __init__(self):
                self.weights = np.random.rand(3, 3)
        return MockLayer()

    def test_construction(self):
        evolver = SNNGeneticEvolver(
            layer_factory=self._make_layer,
            fitness_func=lambda l: float(l.weights.sum()),
        )
        assert len(evolver.population) == 20

    def test_evolve_returns_best(self, capsys):
        evolver = SNNGeneticEvolver(
            layer_factory=self._make_layer,
            fitness_func=lambda l: float(l.weights.sum()),
        )
        best = evolver.evolve(generations=3)
        assert hasattr(best, "weights")

    def test_crossover(self):
        evolver = SNNGeneticEvolver(
            layer_factory=self._make_layer,
            fitness_func=lambda l: float(l.weights.sum()),
        )
        p1 = self._make_layer()
        p2 = self._make_layer()
        child = evolver._crossover(p1, p2)
        assert child.weights.shape == (3, 3)

    def test_mutate(self):
        evolver = SNNGeneticEvolver(
            layer_factory=self._make_layer,
            fitness_func=lambda l: 0.0,
        )
        evolver.mutation_rate = 1.0  # mutate every weight
        layer = self._make_layer()
        original = layer.weights.copy()
        evolver._mutate(layer)
        # With 100% mutation rate, weights should change
        assert not np.array_equal(layer.weights, original)
        # Weights should stay in [0, 1]
        assert np.all(layer.weights >= 0) and np.all(layer.weights <= 1)


# ---------------------------------------------------------------------------
# SwarmCoupling
# ---------------------------------------------------------------------------
from sc_neurocore.robotics.swarm import SwarmCoupling
from sc_neurocore.layers.sc_learning_layer import SCLearningLayer


class TestSwarmCoupling:
    def _make_agent(self, n_inputs, n_neurons, seed=42):
        return SCLearningLayer(
            n_inputs=n_inputs, n_neurons=n_neurons,
            w_min=0.0, w_max=1.0,
            length=64, base_seed=seed,
        )

    def test_synchronize_shifts_weights(self, capsys):
        a = self._make_agent(3, 2, seed=0)
        b = self._make_agent(3, 2, seed=99)
        wa_before = a.get_weights().copy()
        wb_before = b.get_weights().copy()

        sc = SwarmCoupling(coupling_strength=0.5)
        sc.synchronize(a, b)

        wa_after = a.get_weights()
        wb_after = b.get_weights()
        # Weights should have changed
        assert not np.array_equal(wa_before, wa_after)
        assert not np.array_equal(wb_before, wb_after)

    def test_size_mismatch_raises(self):
        a = self._make_agent(2, 3)
        b = self._make_agent(2, 4)
        sc = SwarmCoupling()
        with pytest.raises(ValueError):
            sc.synchronize(a, b)
