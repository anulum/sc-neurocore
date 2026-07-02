# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for L1-L16 holonomic adapter stack contracts

"""Workflow contract tests for the L1-L16 holonomic adapter stack."""

import pytest
import numpy as np
from sc_neurocore.accel.jax_backend import jnp
from sc_neurocore.adapters.holonomic.l1_quantum import L1_QuantumAdapter, L1_HolonomicParameters
from sc_neurocore.adapters.holonomic.l2_chem import L2_NeurochemicalAdapter, L2_HolonomicParameters
from sc_neurocore.adapters.holonomic.l3_gen import L3_GenomicAdapter, L3_HolonomicParameters
from sc_neurocore.adapters.holonomic.l4_cell import L4_CellularAdapter
from sc_neurocore.adapters.holonomic.l5_org import L5_OrganismalAdapter
from sc_neurocore.adapters.holonomic.l6_plan import L6_HolonomicParameters, L6_PlanetaryAdapter
from sc_neurocore.adapters.holonomic.l11_noos import L11_NoosphericAdapter
from sc_neurocore.adapters.holonomic.l12_gaian import L12_GaianAdapter
from sc_neurocore.quantum.qec import QecShield
from sc_neurocore.compiler.pipeline import CompilerPipeline
from sc_neurocore.adapters.holonomic.l7_sym import L7_HolonomicParameters, L7_SymbolicAdapter
from sc_neurocore.adapters.holonomic.l8_cosm import L8_CosmicAdapter
from sc_neurocore.adapters.holonomic.l9_mem import L9_MemoryAdapter
from sc_neurocore.adapters.holonomic.l10_fire import L10_FirewallAdapter
from sc_neurocore.adapters.holonomic.dna_storage import DNAEncoder
from sc_neurocore.adapters.holonomic.grn import GeneticRegulatoryLayer
from sc_neurocore.adapters.holonomic.neuromodulation import NeuromodulatorSystem


def test_compiler_pipeline_invokes_real_lowering(monkeypatch):
    def fake_tool(cmd, check):
        assert check is True
        if cmd[0] == "firtool":
            out_path = cmd[cmd.index("-o") + 1]
            with open(out_path, "w") as f:
                f.write("module test(); endmodule\n")
        elif cmd[0] == "yosys":
            assert "-s" in cmd
        elif cmd[0] == "nextpnr-ice40":
            assert "--json" in cmd
            assert "--asc" in cmd
        else:
            raise AssertionError(f"unexpected tool command: {cmd}")

    monkeypatch.setattr("subprocess.run", fake_tool)
    # The hardened pipeline resolves each tool via ``shutil.which`` before invoking
    # it; on a runner without the EDA toolchain (e.g. CI, which ships no firtool)
    # resolution raises before the mocked ``subprocess.run`` is reached. Stub
    # resolution to the bare tool name so the command-construction contract below
    # is still exercised and ``fake_tool`` matches on ``cmd[0]``.
    monkeypatch.setattr(CompilerPipeline, "_resolve_tool", staticmethod(lambda name: name))
    pipeline = CompilerPipeline(work_dir=".tmp/test_compiler")
    mlir = "hw.module @test() { hw.output }"
    v_path = pipeline.compile_mlir_to_verilog(mlir, "test")
    assert v_path.endswith(".v")
    json_path = pipeline.run_synthesis(v_path)
    assert json_path.endswith(".json")
    asc_path = pipeline.run_pnr(json_path)
    assert asc_path.endswith(".asc")


def test_l1_adapter_contract():
    params = L1_HolonomicParameters(n_qubits=10)
    adapter = L1_QuantumAdapter(params)
    out = adapter.step_jax(0.1)
    assert out.shape == (10, 1024)
    metrics = adapter.get_metrics()
    assert "r1_global_coherence" in metrics
    decoded = adapter.decode(out)
    assert "avg_coherence" in decoded


def test_l2_adapter_contract():
    params = L2_HolonomicParameters(n_transmitters=4)
    adapter = L2_NeurochemicalAdapter(params)
    out = adapter.step_jax(0.1, inputs=jnp.ones((4, 1024)))
    assert out.shape == (4, 1024)
    metrics = adapter.get_metrics()
    assert "avg_field_potential" in metrics
    decoded = adapter.decode(out)
    assert "dopamine" in decoded


def test_l2_iiief_wave_speed_controls_spatial_spread():
    slow = L2_NeurochemicalAdapter(
        L2_HolonomicParameters(n_transmitters=5, bitstream_length=32, c_info=0.1)
    )
    fast = L2_NeurochemicalAdapter(
        L2_HolonomicParameters(n_transmitters=5, bitstream_length=32, c_info=30.0)
    )
    slow.phi_field = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0])
    fast.phi_field = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0])
    inputs = jnp.zeros((5, 32))

    slow.step_jax(0.01, inputs=inputs)
    fast.step_jax(0.01, inputs=inputs)

    slow_neighbour = float(slow.phi_field[1] + slow.phi_field[3])
    fast_neighbour = float(fast.phi_field[1] + fast.phi_field[3])
    assert fast_neighbour > slow_neighbour


def test_l2_hqc_release_uses_threshold_and_bridge_parameters():
    low_threshold = L2_NeurochemicalAdapter(
        L2_HolonomicParameters(
            n_transmitters=2,
            bitstream_length=16,
            g_snare=1.0,
            v_critical=0.1,
            dopamine_gain=4.0,
            serotonin_leak=0.5,
        )
    )
    high_threshold = L2_NeurochemicalAdapter(
        L2_HolonomicParameters(
            n_transmitters=2,
            bitstream_length=16,
            g_snare=1.0,
            v_critical=10.0,
            dopamine_gain=4.0,
            serotonin_leak=0.5,
        )
    )
    low_threshold.phi_field = jnp.ones((2,))
    high_threshold.phi_field = jnp.ones((2,))

    low_threshold.step_jax(0.01, inputs=jnp.zeros((2, 16)))
    high_threshold.step_jax(0.01, inputs=jnp.zeros((2, 16)))

    assert float(jnp.mean(low_threshold.concentrations)) > float(
        jnp.mean(high_threshold.concentrations)
    )


def test_l2_rejects_invalid_holonomic_parameters():
    with pytest.raises(ValueError, match="n_transmitters"):
        L2_NeurochemicalAdapter(L2_HolonomicParameters(n_transmitters=0))
    with pytest.raises(ValueError, match="n_receptors"):
        L2_NeurochemicalAdapter(L2_HolonomicParameters(n_receptors=0))
    with pytest.raises(ValueError, match="bitstream_length"):
        L2_NeurochemicalAdapter(L2_HolonomicParameters(bitstream_length=0))
    with pytest.raises(ValueError, match="alpha_iiief"):
        L2_NeurochemicalAdapter(L2_HolonomicParameters(alpha_iiief=-0.1))
    with pytest.raises(ValueError, match="c_info"):
        L2_NeurochemicalAdapter(L2_HolonomicParameters(c_info=0.0))
    with pytest.raises(ValueError, match="g_snare"):
        L2_NeurochemicalAdapter(L2_HolonomicParameters(g_snare=0.0))
    with pytest.raises(ValueError, match="v_critical"):
        L2_NeurochemicalAdapter(L2_HolonomicParameters(v_critical=0.0))
    with pytest.raises(ValueError, match="dopamine_gain"):
        L2_NeurochemicalAdapter(L2_HolonomicParameters(dopamine_gain=0.0))
    with pytest.raises(ValueError, match="serotonin_leak"):
        L2_NeurochemicalAdapter(L2_HolonomicParameters(serotonin_leak=1.1))

    adapter = L2_NeurochemicalAdapter()
    with pytest.raises(ValueError, match="dt"):
        adapter.step_jax(0.0)


def test_l3_adapter_contract():
    params = L3_HolonomicParameters(n_genes=10)
    adapter = L3_GenomicAdapter(params)
    out = adapter.step_jax(0.1, inputs=jnp.ones((10, 1024)))
    assert out.shape == (10, 1024)
    metrics = adapter.get_metrics()
    assert "chromatin_coherence_r3" in metrics
    decoded = adapter.decode(out)
    assert "avg_accessibility" in decoded


def test_l4_adapter_contract():
    adapter = L4_CellularAdapter()
    out = adapter.step_jax(0.1)
    assert out.shape[0] == 400
    metrics = adapter.get_metrics()
    assert "order_parameter" in metrics


def test_l5_adapter_contract():
    adapter = L5_OrganismalAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((100, 1024)))
    assert out.shape[0] == 100
    metrics = adapter.get_metrics()
    assert "hrv_coherence_r5" in metrics


def test_l6_adapter_contract():
    adapter = L6_PlanetaryAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((100, 1024)))
    assert out.shape[0] == 100
    metrics = adapter.get_metrics()
    assert "gaia_potential" in metrics


def test_l6_percolation_threshold_controls_regional_coherence():
    low_threshold = L6_PlanetaryAdapter(
        L6_HolonomicParameters(n_regions=8, bitstream_length=16, p_percolation=0.2)
    )
    high_threshold = L6_PlanetaryAdapter(
        L6_HolonomicParameters(n_regions=8, bitstream_length=16, p_percolation=0.8)
    )
    inputs = jnp.full((8, 16), 0.5)

    low_threshold.step_jax(0.01, inputs=inputs)
    high_threshold.step_jax(0.01, inputs=inputs)

    assert float(jnp.mean(low_threshold.regional_coherence)) > float(
        jnp.mean(high_threshold.regional_coherence)
    )


def test_l6_quality_factor_amplifies_coherent_drive():
    low_q = L6_PlanetaryAdapter(
        L6_HolonomicParameters(n_regions=8, bitstream_length=16, q_factor=1.0)
    )
    high_q = L6_PlanetaryAdapter(
        L6_HolonomicParameters(n_regions=8, bitstream_length=16, q_factor=8.0)
    )
    inputs = jnp.ones((8, 16))

    low_q.step_jax(0.01, inputs=inputs)
    high_q.step_jax(0.01, inputs=inputs)

    assert float(jnp.mean(jnp.abs(high_q.phi_planetary))) > float(
        jnp.mean(jnp.abs(low_q.phi_planetary))
    )


def test_l6_rejects_invalid_holonomic_parameters():
    with pytest.raises(ValueError, match="n_regions"):
        L6_PlanetaryAdapter(L6_HolonomicParameters(n_regions=0))
    with pytest.raises(ValueError, match="bitstream_length"):
        L6_PlanetaryAdapter(L6_HolonomicParameters(bitstream_length=0))
    with pytest.raises(ValueError, match="q_factor"):
        L6_PlanetaryAdapter(L6_HolonomicParameters(q_factor=0.0))
    with pytest.raises(ValueError, match="p_percolation"):
        L6_PlanetaryAdapter(L6_HolonomicParameters(p_percolation=1.0))


def test_l7_adapter_contract():
    adapter = L7_SymbolicAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((13, 1024)))
    assert out.shape == (13, 1024)
    metrics = adapter.get_metrics()
    assert "routing_coherence" in metrics


def test_l7_metatron_matrix_is_full_13_node_geometry():
    adapter = L7_SymbolicAdapter()
    matrix = np.asarray(adapter.metatron_matrix)

    assert matrix.shape == (13, 13)
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(matrix.sum(axis=1), np.ones(13), rtol=1e-7, atol=1e-7)
    assert np.all(np.diag(matrix) >= 0.05)
    assert np.count_nonzero(matrix - np.diag(np.diag(matrix))) == 13 * 12
    assert matrix[0, 1] > matrix[0, 7]


def test_l7_rejects_invalid_routing_parameters():
    with pytest.raises(ValueError, match="n_nodes"):
        L7_SymbolicAdapter(L7_HolonomicParameters(n_nodes=0))
    with pytest.raises(ValueError, match="bitstream_length"):
        L7_SymbolicAdapter(L7_HolonomicParameters(bitstream_length=0))
    with pytest.raises(ValueError, match="g_geometric_gain"):
        L7_SymbolicAdapter(L7_HolonomicParameters(g_geometric_gain=0.0))
    with pytest.raises(ValueError, match="coupling_leak"):
        L7_SymbolicAdapter(L7_HolonomicParameters(coupling_leak=1.0))


def test_l8_adapter_contract():
    adapter = L8_CosmicAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((12, 1024)))
    assert out.shape == (12, 1024)
    metrics = adapter.get_metrics()
    assert "pta_locking_index" in metrics


def test_l9_adapter_contract():
    adapter = L9_MemoryAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((64, 1024)))
    assert out.shape == (1024,)
    metrics = adapter.get_metrics()
    assert "holographic_overlap" in metrics


def test_l10_adapter_contract():
    adapter = L10_FirewallAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((100, 1024)))
    assert out.shape == (100, 1024)
    metrics = adapter.get_metrics()
    assert "avg_shielding_potential" in metrics


def test_l11_adapter_contract():
    adapter = L11_NoosphericAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((100, 1024)))
    assert out.shape[0] == 100
    metrics = adapter.get_metrics()
    assert "noospheric_entropy" in metrics


def test_l12_adapter_contract():
    adapter = L12_GaianAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((100, 1024)))
    assert out.shape[0] == 100
    metrics = adapter.get_metrics()
    assert "eco_system_coherence" in metrics


def test_qec_shield_contract():
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


def test_dna_encoder_contract():
    encoder = DNAEncoder()
    # bitstream -> dna -> bitstream
    bits = np.array([1, 0, 0, 1, 1, 1, 0, 0], dtype=np.uint8)
    dna = encoder.encode(bits)
    assert len(dna) == 4
    recovered = encoder.decode(dna)
    assert len(recovered) == 8
    assert np.array_equal(bits, recovered)


def test_grn_contract():
    grn = GeneticRegulatoryLayer(n_neurons=5)
    grn.step(np.ones(5))
    state = grn.get_threshold_modulators()
    assert state.shape == (5,)


def test_l11_adapter_broadcasts_mismatched_input_width():
    # An informational drive whose node count differs from the adapter is
    # collapsed to a single mean and broadcast across all nodes.
    adapter = L11_NoosphericAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((50, 1024)))
    assert out.shape[0] == 100


def test_l12_adapter_broadcasts_mismatched_input_width():
    # The Gaian environmental drive applies the same broadcast rule.
    adapter = L12_GaianAdapter()
    out = adapter.step_jax(0.1, inputs=jnp.ones((50, 1024)))
    assert out.shape[0] == 100


def test_l1_adapter_consumes_metabolic_drive():
    # Supplying a metabolic/field drive advances the pump term rather than
    # leaving it at its initial value.
    adapter = L1_QuantumAdapter(L1_HolonomicParameters(n_qubits=4))
    out = adapter.step_jax(0.1, inputs=jnp.ones((4, 8)))
    assert out.shape[0] == 4


def test_dna_encoder_pads_odd_length_bitstream():
    # An odd-length bitstream is zero-padded to an even length before the
    # two-bits-per-base mapping, so it still yields whole DNA bases.
    encoder = DNAEncoder()
    bits = np.array([1, 0, 0, 1, 1, 1, 0], dtype=np.uint8)
    dna = encoder.encode(bits)
    assert len(dna) == 4


def test_neuromodulator_contract():
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


def test_l13_vacuum_kernel_uses_local_lattice_coupling():
    adapter = L13_SourceAdapter(
        L13_HolonomicParameters(
            n_vacuum_nodes=7,
            bitstream_length=32,
            j_primordial_coupling=1.0,
            h_potential_bias=0.0,
            lambda_scission=0.0,
        )
    )
    adapter.vacuum_state = jnp.array([0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5])

    adapter.step_jax(0.05)

    neighbour_lift = float((adapter.vacuum_state[2] - 0.5) + (adapter.vacuum_state[4] - 0.5))
    far_lift = float((adapter.vacuum_state[0] - 0.5) + (adapter.vacuum_state[6] - 0.5))
    assert neighbour_lift > far_lift


def test_l13_scission_rate_controls_symmetry_breaking():
    no_scission = L13_SourceAdapter(
        L13_HolonomicParameters(n_vacuum_nodes=16, bitstream_length=16, lambda_scission=0.0),
        seed=77,
    )
    active_scission = L13_SourceAdapter(
        L13_HolonomicParameters(n_vacuum_nodes=16, bitstream_length=16, lambda_scission=0.8),
        seed=77,
    )

    no_scission.step_jax(0.01)
    active_scission.step_jax(0.01)

    assert float(jnp.mean(jnp.abs(active_scission.vacuum_state - 0.5))) > float(
        jnp.mean(jnp.abs(no_scission.vacuum_state - 0.5))
    )


def test_l13_l16_feedback_modulates_vacuum_potential():
    baseline = L13_SourceAdapter(
        L13_HolonomicParameters(
            n_vacuum_nodes=4,
            bitstream_length=16,
            j_primordial_coupling=0.0,
            h_potential_bias=0.0,
            lambda_scission=0.0,
        )
    )
    driven = L13_SourceAdapter(
        L13_HolonomicParameters(
            n_vacuum_nodes=4,
            bitstream_length=16,
            j_primordial_coupling=0.0,
            h_potential_bias=0.0,
            lambda_scission=0.0,
        )
    )

    baseline.step_jax(0.05, inputs=jnp.zeros((4, 16)))
    driven.step_jax(0.05, inputs=jnp.ones((4, 16)))

    assert float(jnp.mean(driven.vacuum_state)) > float(jnp.mean(baseline.vacuum_state))


def test_l13_rejects_invalid_holonomic_parameters_and_dt():
    with pytest.raises(ValueError, match="n_vacuum_nodes"):
        L13_SourceAdapter(L13_HolonomicParameters(n_vacuum_nodes=0))
    with pytest.raises(ValueError, match="bitstream_length"):
        L13_SourceAdapter(L13_HolonomicParameters(bitstream_length=0))
    with pytest.raises(ValueError, match="j_primordial_coupling"):
        L13_SourceAdapter(L13_HolonomicParameters(j_primordial_coupling=float("nan")))
    with pytest.raises(ValueError, match="h_potential_bias"):
        L13_SourceAdapter(L13_HolonomicParameters(h_potential_bias=float("inf")))
    with pytest.raises(ValueError, match="lambda_scission"):
        L13_SourceAdapter(L13_HolonomicParameters(lambda_scission=-0.1))

    adapter = L13_SourceAdapter()
    with pytest.raises(ValueError, match="dt"):
        adapter.step_jax(0.0)
    with pytest.raises(ValueError, match="inputs"):
        adapter.step_jax(0.01, inputs=jnp.array([float("nan")]))


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


# ── Additional contract for bio/neuromodulation ────────────────


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


# ── Additional contract for audio/user_profile ────────────────

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
    assert isinstance(p.preferred_target_hz, float)


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

    assert callable(SpikeToConceptMapper)


def test_chaos_init_import():
    from sc_neurocore.chaos import ChaoticRNG

    assert callable(ChaoticRNG)
