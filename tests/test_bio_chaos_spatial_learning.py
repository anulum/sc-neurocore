# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for de-omitted modules: chaos, analysis, physics, robotics, learning, spatial, bio."""
from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.bio.neuromodulation import NeuromodulatorSystem
from sc_neurocore.chaos.rng import ChaoticRNG
from sc_neurocore.analysis.explainability import SpikeToConceptMapper
from sc_neurocore.physics.wolfram_hypergraph import WolframHypergraph
from sc_neurocore.robotics.swarm import SwarmCoupling
from sc_neurocore.learning.neuroevolution import SNNGeneticEvolver
from sc_neurocore.spatial.representations import VoxelGrid, PointCloud
from sc_neurocore.spatial.transformer_3d import SpatialTransformer3D
from sc_neurocore.layers.sc_learning_layer import SCLearningLayer


class TestNeuromodulatorSystem:
    def test_update_levels(self):
        nm = NeuromodulatorSystem()
        nm.update_levels(reward=1.0, stress=0.8)
        assert nm.da_level != 0.5
        assert nm.ne_level != 0.1
        assert 0.1 <= nm.ht_level <= 1.0

    def test_modulate_neuron_all_keys(self):
        nm = NeuromodulatorSystem(da_level=0.8, ht_level=0.6, ne_level=0.3)
        params = {"v_threshold": 1.0, "noise_std": 0.5}
        mod = nm.modulate_neuron(params)
        assert mod["v_threshold"] < 1.0
        assert mod["noise_std"] != 0.5

    def test_modulate_neuron_no_keys(self):
        nm = NeuromodulatorSystem()
        params = {"tau_mem": 10.0}
        assert nm.modulate_neuron(params) == {"tau_mem": 10.0}

    def test_serotonin_clip(self):
        nm = NeuromodulatorSystem(ht_level=0.15)
        nm.update_levels(reward=0.0, stress=1.0)
        assert nm.ht_level >= 0.1


class TestChaoticRNG:
    def test_burn_in_changes_state(self):
        rng = ChaoticRNG()
        assert rng.x != 0.5

    def test_random_shape_and_range(self):
        vals = ChaoticRNG().random(200)
        assert vals.shape == (200,)
        assert np.all((vals >= 0) & (vals <= 1))

    def test_deterministic_same_initial(self):
        a = ChaoticRNG(r=4.0, x=0.3)
        b = ChaoticRNG(r=4.0, x=0.3)
        np.testing.assert_array_equal(a.random(50), b.random(50))

    def test_bitstream_shape(self):
        bits = ChaoticRNG().generate_bitstream(0.5, 100)
        assert bits.shape == (100,)
        assert bits.dtype == np.uint8

    def test_bitstream_extremes(self):
        assert np.all(ChaoticRNG().generate_bitstream(0.0, 100) == 0)
        assert np.all(ChaoticRNG().generate_bitstream(1.0, 100) == 1)


class TestSpikeToConceptMapper:
    def test_active_spikes(self):
        mapper = SpikeToConceptMapper({0: "Motor", 2: "Vision"})
        out = mapper.explain(np.array([1, 0, 1, 0]))
        assert "Motor" in out and "Vision" in out

    def test_no_spikes(self):
        assert "idle" in SpikeToConceptMapper({0: "Motor"}).explain(np.array([0, 0, 0]))

    def test_unknown_index(self):
        assert "Unknown(1)" in SpikeToConceptMapper({0: "Motor"}).explain(np.array([0, 1]))

    def test_empty_concept_map(self):
        out = SpikeToConceptMapper({}).explain(np.array([1, 1]))
        assert "Unknown(0)" in out and "Unknown(1)" in out


class TestWolframHypergraph:
    def test_evolve_changes_edges(self):
        hg = WolframHypergraph(edges=[(0, 1), (1, 2)], max_node_id=2)
        hg.evolve(1)
        assert len(hg.edges) != 2

    def test_max_node_id_increments(self):
        hg = WolframHypergraph(edges=[(0, 1), (1, 2)], max_node_id=2)
        hg.evolve(1)
        assert hg.max_node_id > 2

    def test_dimension_estimate(self):
        hg = WolframHypergraph(edges=[(0, 1), (1, 2)], max_node_id=2)
        assert hg.dimension_estimate() == 2
        hg.evolve(1)
        assert hg.dimension_estimate() == len(hg.edges)

    def test_non_binary_edges_skipped(self):
        hg = WolframHypergraph(edges=[(0, 1, 2), (1, 2), (2, 3)], max_node_id=3)
        hg.evolve(1)
        assert any(len(e) == 3 for e in hg.edges)

    def test_multi_step(self):
        hg = WolframHypergraph(edges=[(0, 1), (1, 2), (2, 3), (3, 4)], max_node_id=4)
        hg.evolve(3)
        assert len(hg.edges) > 0


class TestSwarmCoupling:
    @pytest.fixture()
    def agents(self):
        a = SCLearningLayer(n_inputs=4, n_neurons=3, base_seed=42)
        b = SCLearningLayer(n_inputs=4, n_neurons=3, base_seed=99)
        return a, b

    def test_synchronize_shifts_weights(self, agents):
        a, b = agents
        wa_before = a.get_weights().copy()
        SwarmCoupling(coupling_strength=0.5).synchronize(a, b)
        assert not np.array_equal(wa_before, a.get_weights())

    def test_mismatched_raises(self):
        a = SCLearningLayer(n_inputs=4, n_neurons=3, base_seed=1)
        b = SCLearningLayer(n_inputs=4, n_neurons=5, base_seed=2)
        with pytest.raises(ValueError, match="same size"):
            SwarmCoupling().synchronize(a, b)

    def test_zero_coupling_no_change(self, agents):
        a, b = agents
        wa_before = a.get_weights().copy()
        SwarmCoupling(coupling_strength=0.0).synchronize(a, b)
        np.testing.assert_array_equal(wa_before, a.get_weights())


class _Individual:
    def __init__(self):
        self.weights = np.random.rand(4, 4)


class TestSNNGeneticEvolver:
    def test_population_size(self):
        evo = SNNGeneticEvolver(lambda: _Individual(), lambda ind: float(np.sum(ind.weights)))
        assert len(evo.population) == 20

    def test_evolve_returns_best(self):
        evo = SNNGeneticEvolver(lambda: _Individual(), lambda ind: float(np.sum(ind.weights)))
        best = evo.evolve(3)
        assert hasattr(best, "weights")

    def test_crossover_mixes(self):
        evo = SNNGeneticEvolver(lambda: _Individual(), lambda ind: 0.0)
        p1, p2 = _Individual(), _Individual()
        p1.weights, p2.weights = np.zeros((4, 4)), np.ones((4, 4))
        child = evo._crossover(p1, p2)
        assert child.weights.shape == (4, 4)

    def test_mutate_within_bounds(self):
        evo = SNNGeneticEvolver(lambda: _Individual(), lambda ind: 0.0)
        ind = _Individual()
        ind.weights = np.full((4, 4), 0.5)
        evo._mutate(ind)
        assert np.all(ind.weights >= 0) and np.all(ind.weights <= 1)

    def test_crossover_no_weights(self):
        evo = SNNGeneticEvolver(lambda: object(), lambda ind: 0.0)
        child = evo._crossover(object(), object())
        assert not hasattr(child, "weights")

    def test_mutate_no_weights(self):
        evo = SNNGeneticEvolver(lambda: object(), lambda ind: 0.0)
        evo._mutate(object())


class TestVoxelGrid:
    def test_init_zeros(self):
        vg = VoxelGrid(resolution=4)
        assert vg.data.shape == (4, 4, 4)
        assert np.all(vg.data == 0)

    def test_set_voxel(self):
        vg = VoxelGrid(resolution=4)
        vg.set_voxel(1, 2, 3, 0.9)
        assert vg.data[1, 2, 3] == 0.9

    def test_set_voxel_out_of_bounds(self):
        vg = VoxelGrid(resolution=4)
        vg.set_voxel(10, 10, 10, 1.0)
        assert np.all(vg.data == 0)

    def test_bitstream_shape(self):
        bs = VoxelGrid(resolution=2).get_as_bitstream(length=64)
        assert bs.shape == (2, 2, 2, 64)
        assert bs.dtype == np.uint8


class TestPointCloud:
    def test_normalize(self):
        pc = PointCloud(
            points=np.array([[0.0, 10.0, 20.0], [5.0, 15.0, 25.0]]),
            intensities=np.array([0.5, 1.5]),
        )
        pc.normalize()
        assert np.min(pc.points) >= 0.0
        assert np.max(pc.points) <= 1.0 + 1e-9
        assert np.all(pc.intensities <= 1.0)


class TestSpatialTransformer3D:
    def test_output_shape(self):
        grid = np.random.rand(3, 3, 3)
        out = SpatialTransformer3D(resolution=3, dim_k=4).forward(grid)
        assert out.shape == (3, 3, 3)

    def test_output_differs(self):
        grid = np.random.rand(3, 3, 3)
        out = SpatialTransformer3D(resolution=3, dim_k=4).forward(grid)
        assert not np.array_equal(grid, out)
