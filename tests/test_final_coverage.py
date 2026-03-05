# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests to close ALL remaining coverage gaps from 98% → 100%."""

import numpy as np
import pytest


# ── ensembles/orchestrator.py — run_consensus loop body (lines 22-28) ─
class TestEnsembleRunConsensus:
    def test_run_consensus_with_pipeline(self):
        """Execute run_consensus with real agents that have forward() modules."""
        from sc_neurocore.ensembles.orchestrator import EnsembleOrchestrator
        from sc_neurocore.core.orchestrator import CognitiveOrchestrator
        from sc_neurocore.core.tensor_stream import TensorStream
        from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer

        agent1 = CognitiveOrchestrator()
        layer1 = VectorizedSCLayer(n_inputs=2, n_neurons=2, length=64)
        agent1.register_module("sc", layer1)

        agent2 = CognitiveOrchestrator()
        layer2 = VectorizedSCLayer(n_inputs=2, n_neurons=2, length=64)
        agent2.register_module("sc", layer2)

        ens = EnsembleOrchestrator()
        ens.add_agent("a1", agent1)
        ens.add_agent("a2", agent2)

        initial = TensorStream.from_prob(np.array([0.5, 0.5]))
        result = ens.run_consensus(["sc"], initial)
        assert isinstance(result, np.ndarray)
        assert len(result) == 2


# ── generative/three_d_gen.py — empty normals and subsample ──────────
class TestThreeDGenBranches:
    def test_compute_normals_empty(self):
        """Hit line 213: _compute_normals with empty vertices."""
        from sc_neurocore.generative.three_d_gen import SC3DGenerator

        g = SC3DGenerator()
        normals = g._compute_normals(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32))
        assert normals.shape == (0, 3)

    def test_bitstream_to_voxels_subsample(self):
        """Hit lines 312-313: n_units >= n_voxels → subsample path."""
        from sc_neurocore.generative.three_d_gen import SC3DGenerator

        g = SC3DGenerator()
        # 100 units > 8 voxels → subsample
        bs = np.random.randint(0, 2, (100, 32)).astype(np.uint8)
        voxels = g.bitstream_to_voxels(bs, grid_size=(2, 2, 2))
        assert voxels.shape == (2, 2, 2)


# ── pipeline/ingestion.py — constant data (line 35) ──────────────────
class TestIngestionConstant:
    def test_constant_data(self):
        """arr_max == arr_min → zeros_like."""
        from sc_neurocore.pipeline.ingestion import DataIngestor

        ingestor = DataIngestor()
        ds = ingestor.prepare_dataset({"v": [5.0, 5.0, 5.0]})
        assert np.all(ds.data["v"] == 0)


# ── pipeline/training.py — RL with RSTDP synapse (line 31) ──────────
class TestTrainingRLWithRSTDP:
    def test_rl_epoch_rstdp(self):
        """Replace synapses with RewardModulatedSTDPSynapse to hit line 31."""
        from sc_neurocore.pipeline.training import SCTrainingLoop
        from sc_neurocore.layers.sc_learning_layer import SCLearningLayer
        from sc_neurocore.synapses.r_stdp import RewardModulatedSTDPSynapse

        agent = SCLearningLayer(n_inputs=2, n_neurons=1, length=32)
        # Replace synapses with RSTDP
        for i in range(agent.n_neurons):
            for j in range(agent.n_inputs):
                old = agent.synapses[i][j]
                agent.synapses[i][j] = RewardModulatedSTDPSynapse(
                    w_min=old.w_min,
                    w_max=old.w_max,
                    w=old.w,
                    learning_rate=old.learning_rate,
                    length=old.length,
                )
        data = np.array([0.5, 0.5])
        SCTrainingLoop.run_rl_epoch(
            agent=agent,
            env_step_func=lambda spikes: float(np.sum(spikes)),
            input_data=data,
            generations=2,
        )


# ── utils/model_bridge.py — load with synapses (lines 54-56) ────────
class TestModelBridgeSynapses:
    def test_load_into_layer_with_synapses(self):
        """Layer has both .weights and .synapses → hit lines 54-56."""
        from sc_neurocore.utils.model_bridge import SCBridge
        from sc_neurocore.layers.sc_learning_layer import SCLearningLayer

        layer = SCLearningLayer(n_inputs=2, n_neurons=2, length=32)
        # Add a .weights attribute so load_from_state_dict can match it
        layer.weights = layer.get_weights()
        SCBridge.load_from_state_dict(
            {"l.weight": layer.weights.copy()},
            {"l": layer},
        )


# ── verification/formal_proofs.py — energy < cost (lines 59-60) ─────
class TestFormalVerifierEnergy:
    def test_energy_insufficient(self):
        from sc_neurocore.verification.formal_proofs import FormalVerifier

        result = FormalVerifier.verify_energy_safety(energy=1.0, cost=5.0)
        assert result is False


# ── SCPN L2 — history overflow, release_nt, neuromodulation ─────────
class TestL2DeepBranches:
    def test_history_overflow(self):
        """Hit line 147: history.pop(0) when len > 100."""
        from sc_neurocore.scpn.layers.l2_neurochemical import (
            L2_NeurochemicalLayer,
            L2_StochasticParameters,
        )

        params = L2_StochasticParameters(n_receptors=4, bitstream_length=32)
        layer = L2_NeurochemicalLayer(params)
        for _ in range(105):
            layer.step(0.01)
        assert len(layer.history) <= 100

    def test_release_neurotransmitter(self):
        """Hit lines 158-159: valid nt_type release."""
        from sc_neurocore.scpn.layers.l2_neurochemical import (
            L2_NeurochemicalLayer,
            L2_StochasticParameters,
        )

        params = L2_StochasticParameters(n_receptors=4, bitstream_length=32)
        layer = L2_NeurochemicalLayer(params)
        layer.release_neurotransmitter(0, 0.5)
        assert layer.nt_concentrations[0] > 0

    def test_get_neuromodulation_state(self):
        """Hit line 169: get_neuromodulation_state."""
        from sc_neurocore.scpn.layers.l2_neurochemical import (
            L2_NeurochemicalLayer,
            L2_StochasticParameters,
        )

        params = L2_StochasticParameters(n_receptors=4, bitstream_length=32)
        layer = L2_NeurochemicalLayer(params)
        layer.step(0.01)
        state = layer.get_neuromodulation_state()
        assert "dopamine" in state
        assert "serotonin" in state


# ── SCPN L3 — get_ciss_coherence ────────────────────────────────────
class TestL3DeepBranches:
    def test_ciss_coherence(self):
        """Hit line 199: get_ciss_coherence."""
        from sc_neurocore.scpn.layers.l3_genomic import (
            L3_GenomicLayer,
            L3_StochasticParameters,
        )

        params = L3_StochasticParameters(n_genes=4, bitstream_length=32)
        layer = L3_GenomicLayer(params)
        layer.step(0.01)
        c = layer.get_ciss_coherence()
        assert isinstance(c, float)


# ── SCPN L4 — get_tissue_pattern ────────────────────────────────────
class TestL4DeepBranches:
    def test_tissue_pattern(self):
        """Hit line 202: get_tissue_pattern."""
        from sc_neurocore.scpn.layers.l4_cellular import (
            L4_CellularLayer,
            L4_StochasticParameters,
        )

        params = L4_StochasticParameters(grid_size=(4, 4), bitstream_length=32)
        layer = L4_CellularLayer(params)
        layer.step(0.01)
        pattern = layer.get_tissue_pattern()
        assert pattern.shape == (4, 4)


# ── SCPN L5 — all uncovered branches ────────────────────────────────
class TestL5DeepBranches:
    def test_external_event_int_keys(self):
        """Hit line 130: external_event with int keys."""
        from sc_neurocore.scpn.layers.l5_organismal import (
            L5_OrganismalLayer,
            L5_StochasticParameters,
        )

        params = L5_StochasticParameters(
            n_emotional_dims=8,
            n_autonomic_nodes=16,
            bitstream_length=32,
        )
        layer = L5_OrganismalLayer(params)
        output = layer.step(0.01, external_event={0: 0.5, 2: 0.3})
        assert output is not None

    def test_rmssd_with_intervals(self):
        """Hit lines 233-235: _compute_rmssd when rr_intervals >= 2."""
        from sc_neurocore.scpn.layers.l5_organismal import (
            L5_OrganismalLayer,
            L5_StochasticParameters,
        )

        params = L5_StochasticParameters(
            n_emotional_dims=8,
            n_autonomic_nodes=16,
            bitstream_length=32,
        )
        layer = L5_OrganismalLayer(params)
        # Need at least 2 steps to accumulate 2 rr_intervals
        layer.step(0.01)
        layer.step(0.01)
        rmssd = layer._compute_rmssd()
        assert isinstance(rmssd, float)
        assert rmssd >= 0

    def test_rr_intervals_overflow(self):
        """Hit line 190: rr_intervals.pop(0) when len > 100."""
        from sc_neurocore.scpn.layers.l5_organismal import (
            L5_OrganismalLayer,
            L5_StochasticParameters,
        )

        params = L5_StochasticParameters(
            n_emotional_dims=8,
            n_autonomic_nodes=16,
            bitstream_length=32,
        )
        layer = L5_OrganismalLayer(params)
        for _ in range(105):
            layer.step(0.01)
        assert len(layer.rr_intervals) <= 100

    def test_get_emotional_valence(self):
        """Hit line 246: get_emotional_valence."""
        from sc_neurocore.scpn.layers.l5_organismal import (
            L5_OrganismalLayer,
            L5_StochasticParameters,
        )

        params = L5_StochasticParameters(
            n_emotional_dims=8,
            n_autonomic_nodes=16,
            bitstream_length=32,
        )
        layer = L5_OrganismalLayer(params)
        layer.step(0.01)
        valence = layer.get_emotional_valence()
        assert isinstance(valence, float)


# ── SCPN L6 — all uncovered branches ────────────────────────────────
class TestL6DeepBranches:
    def test_history_overflow(self):
        """Hit line 218: history.pop(0) when len > 100."""
        from sc_neurocore.scpn.layers.l6_ecological import (
            L6_EcologicalLayer,
            L6_StochasticParameters,
        )

        params = L6_StochasticParameters(n_field_nodes=16, bitstream_length=32)
        layer = L6_EcologicalLayer(params)
        for _ in range(105):
            layer.step(0.01)
        assert len(layer.history) <= 100

    def test_get_schumann_spectrum(self):
        """Hit line 228: get_schumann_spectrum."""
        from sc_neurocore.scpn.layers.l6_ecological import (
            L6_EcologicalLayer,
            L6_StochasticParameters,
        )

        params = L6_StochasticParameters(n_field_nodes=16, bitstream_length=32)
        layer = L6_EcologicalLayer(params)
        layer.step(0.01)
        spectrum = layer.get_schumann_spectrum()
        assert isinstance(spectrum, dict)
        assert len(spectrum) > 0

    def test_get_circadian_time(self):
        """Hit line 239: get_circadian_time."""
        from sc_neurocore.scpn.layers.l6_ecological import (
            L6_EcologicalLayer,
            L6_StochasticParameters,
        )

        params = L6_StochasticParameters(n_field_nodes=16, bitstream_length=32)
        layer = L6_EcologicalLayer(params)
        layer.step(0.01)
        ct = layer.get_circadian_time()
        assert isinstance(ct, float)
        assert 0 <= ct <= 24


# ── SCPN L7 — all uncovered branches ────────────────────────────────
class TestL7DeepBranches:
    def test_fallback_phi_and_fib_and_e8(self):
        """Hit lines 139, 150, 180: zero activations → fallback values."""
        from sc_neurocore.scpn.layers.l7_symbolic import (
            L7_SymbolicLayer,
            L7_StochasticParameters,
        )

        params = L7_StochasticParameters(n_symbols=32, bitstream_length=32)
        layer = L7_SymbolicLayer(params)
        # Force near-zero activations so sorted[1] <= 0.01 and e8_norm ≈ 0
        layer.symbol_activations = np.zeros(params.n_symbols)
        layer.e8_state = np.zeros(8)
        output = layer.step(0.01)
        assert output["phi_alignment"] == 0.5
        assert output["fibonacci_alignment"] == 0.5
        assert output["e8_alignment"] == 0.5

    def test_get_glyph_vector_normalized(self):
        """Hit line 270: get_glyph_vector_normalized."""
        from sc_neurocore.scpn.layers.l7_symbolic import (
            L7_SymbolicLayer,
            L7_StochasticParameters,
        )

        params = L7_StochasticParameters(n_symbols=32, bitstream_length=32)
        layer = L7_SymbolicLayer(params)
        layer.step(0.01)
        glyph = layer.get_glyph_vector_normalized()
        assert isinstance(glyph, np.ndarray)
        assert len(glyph) == params.glyph_dimensions

    def test_stimulate_meridian(self):
        """Hit lines 274-275: stimulate_meridian with valid id."""
        from sc_neurocore.scpn.layers.l7_symbolic import (
            L7_SymbolicLayer,
            L7_StochasticParameters,
        )

        params = L7_StochasticParameters(n_symbols=32, bitstream_length=32)
        layer = L7_SymbolicLayer(params)
        old_qi = layer.meridian_qi[0]
        layer.stimulate_meridian(0, 0.5)
        assert layer.meridian_qi[0] >= old_qi

    def test_get_acupoint_map(self):
        """Hit lines 283-293: get_acupoint_map."""
        from sc_neurocore.scpn.layers.l7_symbolic import (
            L7_SymbolicLayer,
            L7_StochasticParameters,
        )

        # Use default n_acupoints=361 so most named points are valid
        params = L7_StochasticParameters(n_symbols=32, bitstream_length=32)
        layer = L7_SymbolicLayer(params)
        layer.step(0.01)
        acu_map = layer.get_acupoint_map()
        assert isinstance(acu_map, dict)
        assert "LI4_Hegu" in acu_map
