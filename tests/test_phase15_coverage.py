# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Comprehensive Unit Tests for Phase 15 and 16 logic to ensure 97%+ CI coverage.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from sc_neurocore.adapters.holonomic.l1_quantum import L1_QuantumAdapter, L1_HolonomicParameters
from sc_neurocore.adapters.holonomic.l2_chem import L2_NeurochemicalAdapter, L2_HolonomicParameters
from sc_neurocore.adapters.holonomic.l3_gen import L3_GenomicAdapter, L3_HolonomicParameters
from sc_neurocore.adapters.holonomic.l4_cell import L4_CellularAdapter, L4_HolonomicParameters
from sc_neurocore.adapters.holonomic.l5_org import L5_OrganismalAdapter, L5_HolonomicParameters
from sc_neurocore.adapters.holonomic.l6_plan import L6_PlanetaryAdapter, L6_HolonomicParameters
from sc_neurocore.adapters.holonomic.l11_noos import L11_NoosphericAdapter, L11_HolonomicParameters
from sc_neurocore.adapters.holonomic.l12_gaian import L12_GaianAdapter, L12_HolonomicParameters
from sc_neurocore.quantum.qec import QecShield
from sc_neurocore.compiler.pipeline import CompilerPipeline
from sc_neurocore.adapters.holonomic.l7_sym import L7_SymbolicAdapter, L7_HolonomicParameters
from sc_neurocore.adapters.holonomic.l8_cosm import L8_CosmicAdapter, L8_HolonomicParameters
from sc_neurocore.adapters.holonomic.l9_mem import L9_MemoryAdapter, L9_HolonomicParameters
from sc_neurocore.adapters.holonomic.l10_fire import L10_FirewallAdapter, L10_HolonomicParameters
from sc_neurocore.adapters.holonomic.dna_storage import DNAEncoder
from sc_neurocore.adapters.holonomic.grn import GeneticRegulatoryLayer
from sc_neurocore.adapters.holonomic.neuromodulation import NeuromodulatorSystem


def test_compiler_pipeline_stubs():
    pipeline = CompilerPipeline(work_dir=".tmp/test_compiler")
    mlir = "hw.module @test() { hw.output }"
    v_path = pipeline.compile_mlir_to_verilog(mlir, "test")
    assert v_path.endswith(".v")
    json_path = pipeline.run_synthesis(v_path)
    assert json_path.endswith(".json")
    asc_path = pipeline.run_pnr(json_path)
    assert asc_path.endswith(".asc")


def test_l1_adapter_coverage():
    params = L1_HolonomicParameters(n_qubits=10)
    adapter = L1_QuantumAdapter(params)
    out = adapter.step_jax(0.1)
    assert out.shape == (10, 1024)
    metrics = adapter.get_metrics()
    assert "r1_global_coherence" in metrics
    decoded = adapter.decode(out)
    assert "avg_coherence" in decoded


def test_l2_adapter_coverage():
    params = L2_HolonomicParameters(n_transmitters=4)
    adapter = L2_NeurochemicalAdapter(params)
    out = adapter.step_jax(0.1, inputs=jnp.ones((4, 1024)))
    assert out.shape == (4, 1024)
    metrics = adapter.get_metrics()
    assert "avg_field_potential" in metrics
    decoded = adapter.decode(out)
    assert "dopamine" in decoded


def test_l3_adapter_coverage():
    params = L3_HolonomicParameters(n_genes=10)
    adapter = L3_GenomicAdapter(params)
    out = adapter.step_jax(0.1, inputs=jnp.ones((10, 1024)))
    assert out.shape == (10, 1024)
    metrics = adapter.get_metrics()
    assert "chromatin_coherence_r3" in metrics
    decoded = adapter.decode(out)
    assert "avg_accessibility" in decoded


def test_l4_adapter_coverage():
    adapter = L4_CellularAdapter()
    out = adapter.step_jax(0.1)
    assert out.shape[0] == 400
    metrics = adapter.get_metrics()
    assert "order_parameter" in metrics


def test_l5_adapter_coverage():
    adapter = L5_OrganismalAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((100, 1024)))
    assert out.shape[0] == 100
    metrics = adapter.get_metrics()
    assert "hrv_coherence_r5" in metrics


def test_l6_adapter_coverage():
    adapter = L6_PlanetaryAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((100, 1024)))
    assert out.shape[0] == 100
    metrics = adapter.get_metrics()
    assert "gaia_potential" in metrics


def test_l7_adapter_coverage():
    adapter = L7_SymbolicAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((13, 1024)))
    assert out.shape == (13, 1024)
    metrics = adapter.get_metrics()
    assert "routing_coherence" in metrics


def test_l8_adapter_coverage():
    adapter = L8_CosmicAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((12, 1024)))
    assert out.shape == (12, 1024)
    metrics = adapter.get_metrics()
    assert "pta_locking_index" in metrics


def test_l9_adapter_coverage():
    adapter = L9_MemoryAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((64, 1024)))
    assert out.shape == (1024,)
    metrics = adapter.get_metrics()
    assert "holographic_overlap" in metrics


def test_l10_adapter_coverage():
    adapter = L10_FirewallAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((100, 1024)))
    assert out.shape == (100, 1024)
    metrics = adapter.get_metrics()
    assert "avg_shielding_potential" in metrics


def test_l11_adapter_coverage():
    adapter = L11_NoosphericAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((100, 1024)))
    assert out.shape[0] == 100
    metrics = adapter.get_metrics()
    assert "noospheric_entropy" in metrics


def test_l12_adapter_coverage():
    adapter = L12_GaianAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((100, 1024)))
    assert out.shape[0] == 100
    metrics = adapter.get_metrics()
    assert "eco_system_coherence" in metrics


def test_qec_shield_coverage():
    shield = QecShield(code_type="repetition", distance=3)
    bits = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    encoded = shield.encode(bits)
    assert encoded.shape == (2, 3, 2)
    syndromes = shield.extract_syndromes(encoded)
    assert syndromes.shape == (2, 2, 2)
    decoded = shield.decode(encoded)
    assert np.array_equal(decoded, bits)
    rate = shield.get_error_rate(syndromes)
    assert rate == 0.0


def test_dna_encoder_coverage():
    encoder = DNAEncoder()
    # bitstream -> dna -> bitstream
    bits = np.array([1, 0, 0, 1, 1, 1, 0, 0], dtype=np.uint8)
    dna = encoder.encode(bits)
    assert len(dna) == 4
    recovered = encoder.decode(dna)
    assert len(recovered) == 8
    assert np.array_equal(bits, recovered)


def test_grn_coverage():
    grn = GeneticRegulatoryLayer(n_neurons=5)
    grn.step(np.ones(5))
    state = grn.get_threshold_modulators()
    assert state.shape == (5,)


def test_neuromodulator_coverage():
    mod = NeuromodulatorSystem()
    mod.update_levels(reward=0.8, stress=0.2)
    params = {"v_threshold": 1.0, "noise_std": 0.1}
    new_params = mod.modulate_neuron(params)
    assert new_params["v_threshold"] < 1.0


# ── L13-L16 Adapter Tests (Phase 16) ──────────────────────────

from sc_neurocore.adapters.holonomic.l13_source import L13_SourceAdapter, L13_HolonomicParameters
from sc_neurocore.adapters.holonomic.l14_trans import (
    L14_TransdimensionalAdapter,
    L14_HolonomicParameters,
)
from sc_neurocore.adapters.holonomic.l15_cons import L15_ConsiliumAdapter, L15_HolonomicParameters
from sc_neurocore.adapters.holonomic.l16_meta import L16_MetaAdapter, L16_HolonomicParameters
from sc_neurocore.adapters.base import BaseStochasticAdapter


def test_l13_adapter_init_and_step():
    params = L13_HolonomicParameters(n_vacuum_nodes=8, bitstream_length=64)
    adapter = L13_SourceAdapter(params)
    out = adapter.step_jax(0.1)
    assert out.shape == (8, 64)
    metrics = adapter.get_metrics()
    assert "vacuum_potential" in metrics
    assert "fisher_information_metric" in metrics
    decoded = adapter.decode(out)
    assert "source_coherence_r13" in decoded


def test_l13_encode():
    adapter = L13_SourceAdapter(L13_HolonomicParameters(n_vacuum_nodes=4, bitstream_length=32))
    bits = adapter.encode(None)
    assert bits.shape == (4, 32)
    assert bits.dtype == jnp.uint8


def test_l13_vacuum_kernel_clip():
    adapter = L13_SourceAdapter(L13_HolonomicParameters(n_vacuum_nodes=4))
    for _ in range(50):
        adapter.step_jax(0.5)
    state = adapter.vacuum_state
    assert float(jnp.min(state)) >= 0.0
    assert float(jnp.max(state)) <= 1.0


def test_l14_adapter_init_and_step():
    params = L14_HolonomicParameters(n_bulk_dimensions=5, bitstream_length=64)
    adapter = L14_TransdimensionalAdapter(params)
    out = adapter.step_jax(0.1)
    assert out.shape == (5, 64)
    metrics = adapter.get_metrics()
    assert "avg_brane_alignment" in metrics
    assert "resonance_sharpness" in metrics
    decoded = adapter.decode(out)
    assert "brane_resonance_r14" in decoded


def test_l14_with_inputs():
    adapter = L14_TransdimensionalAdapter(
        L14_HolonomicParameters(n_bulk_dimensions=5, bitstream_length=32)
    )
    inputs = jnp.ones((5, 32))
    out = adapter.step_jax(0.1, inputs=inputs)
    assert out.shape == (5, 32)


def test_l14_input_broadcast():
    adapter = L14_TransdimensionalAdapter(
        L14_HolonomicParameters(n_bulk_dimensions=5, bitstream_length=32)
    )
    inputs = jnp.ones((3, 32))
    out = adapter.step_jax(0.1, inputs=inputs)
    assert out.shape == (5, 32)


def test_l15_adapter_init_and_step():
    params = L15_HolonomicParameters(n_metric_dimensions=8, bitstream_length=64)
    adapter = L15_ConsiliumAdapter(params)
    out = adapter.step_jax(0.1)
    assert out.shape == (8, 64)
    metrics = adapter.get_metrics()
    assert "gci_index" in metrics
    assert "metric_entropy" in metrics
    assert "optimizer_error" in metrics
    decoded = adapter.decode(out)
    assert "global_coherence_r15" in decoded


def test_l15_with_inputs():
    adapter = L15_ConsiliumAdapter(
        L15_HolonomicParameters(n_metric_dimensions=8, bitstream_length=32)
    )
    inputs = jnp.ones((8, 32))
    out = adapter.step_jax(0.1, inputs=inputs)
    assert out.shape == (8, 32)


def test_l15_partial_stack_padding():
    adapter = L15_ConsiliumAdapter(
        L15_HolonomicParameters(n_metric_dimensions=16, bitstream_length=32)
    )
    inputs = jnp.ones((4, 32))
    out = adapter.step_jax(0.1, inputs=inputs)
    assert out.shape == (16, 32)


def test_l16_adapter_init_and_step():
    params = L16_HolonomicParameters(n_meta_nodes=4, bitstream_length=64)
    adapter = L16_MetaAdapter(params)
    out = adapter.step_jax(0.1)
    assert out.shape == (4, 64)
    metrics = adapter.get_metrics()
    assert "director_will" in metrics
    assert "system_entropy" in metrics
    assert "veto_active" in metrics
    decoded = adapter.decode(out)
    assert "meta_coherence_r16" in decoded


def test_l16_with_gci_input():
    adapter = L16_MetaAdapter(L16_HolonomicParameters(n_meta_nodes=4, bitstream_length=32))
    inputs = jnp.ones((1, 32))
    out = adapter.step_jax(0.1, inputs=inputs)
    assert out.shape == (4, 32)


def test_l16_veto_activation():
    params = L16_HolonomicParameters(n_meta_nodes=4, veto_threshold=0.1)
    adapter = L16_MetaAdapter(params)
    adapter.entropy_proxy = 0.9
    adapter.step_jax(0.1)
    assert float(jnp.mean(adapter.veto_active)) > 0.0


def test_base_adapter_is_abstract():
    with pytest.raises(TypeError):
        BaseStochasticAdapter()


# ── Additional coverage for bio/neuromodulation ────────────────


def test_neuromodulator_stress_clamp():
    mod = NeuromodulatorSystem()
    mod.update_levels(reward=0.0, stress=10.0)
    assert mod.ht_level >= 0.1
    mod.update_levels(reward=0.0, stress=-10.0)
    assert mod.ht_level <= 1.0


def test_neuromodulator_no_matching_keys():
    mod = NeuromodulatorSystem()
    mod.update_levels(reward=0.5, stress=0.5)
    params = {"some_other_param": 42}
    out = mod.modulate_neuron(params)
    assert out["some_other_param"] == 42


# ── Additional coverage for audio/user_profile ────────────────

from sc_neurocore.audio.user_profile import UserProfile, Chronotype


def test_user_profile_defaults():
    p = UserProfile()
    assert p.chronotype == Chronotype.BEAR
    assert p.get_best_target_hz() == 10.0
    assert "alpha" in p.sensitivity_map


def test_user_profile_explicit_target():
    p = UserProfile(preferred_target_hz=7.5)
    assert p.get_best_target_hz() == 7.5


def test_user_profile_update_session():
    p = UserProfile(chronotype=Chronotype.WOLF)
    p.update_from_session(avg_evs=60.0, peak_evs=80.0, best_target_hz=8.0)
    assert p.session_count == 1
    assert p.preferred_target_hz == 8.0
    p.update_from_session(avg_evs=55.0, peak_evs=70.0, best_target_hz=9.0)
    assert p.session_count == 2
    assert p.preferred_target_hz is not None


def test_user_profile_update_low_evs():
    p = UserProfile()
    p.update_from_session(avg_evs=30.0, peak_evs=40.0, best_target_hz=5.0)
    assert p.preferred_target_hz is None


def test_user_profile_update_band_powers():
    p = UserProfile()
    p.update_from_session(avg_evs=60.0, peak_evs=70.0, band_powers={"alpha": 10.0})
    assert "alpha" in p.baseline_band_powers
    p.update_from_session(avg_evs=60.0, peak_evs=70.0, band_powers={"alpha": 20.0})
    assert p.baseline_band_powers["alpha"] != 10.0


def test_user_profile_serialization():
    p = UserProfile(user_id="test-1", chronotype=Chronotype.LION)
    d = p.to_dict()
    p2 = UserProfile.from_dict(d)
    assert p2.user_id == "test-1"
    assert p2.chronotype == Chronotype.LION


def test_user_profile_all_chronotypes():
    for chrono in Chronotype:
        p = UserProfile(chronotype=chrono)
        assert p.get_best_target_hz() > 0


# ── analysis & chaos __init__ imports ──────────────────────────


def test_analysis_init_import():
    from sc_neurocore.analysis import SpikeToConceptMapper

    assert SpikeToConceptMapper is not None


def test_chaos_init_import():
    from sc_neurocore.chaos import ChaoticRNG

    assert ChaoticRNG is not None
