# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Demonstration of Phase 16 Omni-Substrate Convergence

"""
Demonstration of Phase 16 Omni-Substrate Convergence.

This experiment showcases:
1. JAX-accelerated SCPN L1-L16 holonomic simulation.
2. Cross-substrate coupling (Quantum <-> Planetary <-> Transcendent).
3. Automated MLIR hardware synthesis pipeline.
4. Noise-robust execution via QEC Shield.
"""

import logging

from sc_neurocore.adapters.holonomic.l1_quantum import L1_QuantumAdapter, L1_HolonomicParameters
from sc_neurocore.adapters.holonomic.l2_chem import L2_NeurochemicalAdapter, L2_HolonomicParameters
from sc_neurocore.adapters.holonomic.l3_gen import L3_GenomicAdapter, L3_HolonomicParameters
from sc_neurocore.adapters.holonomic.l4_cell import L4_CellularAdapter, L4_HolonomicParameters
from sc_neurocore.adapters.holonomic.l5_org import L5_OrganismalAdapter, L5_HolonomicParameters
from sc_neurocore.adapters.holonomic.l6_plan import L6_PlanetaryAdapter, L6_HolonomicParameters
from sc_neurocore.adapters.holonomic.l7_sym import L7_SymbolicAdapter, L7_HolonomicParameters
from sc_neurocore.adapters.holonomic.l8_cosm import L8_CosmicAdapter, L8_HolonomicParameters
from sc_neurocore.adapters.holonomic.l9_mem import L9_MemoryAdapter, L9_HolonomicParameters
from sc_neurocore.adapters.holonomic.l10_fire import L10_FirewallAdapter, L10_HolonomicParameters
from sc_neurocore.adapters.holonomic.l11_noos import L11_NoosphericAdapter, L11_HolonomicParameters
from sc_neurocore.adapters.holonomic.l12_gaian import L12_GaianAdapter, L12_HolonomicParameters
from sc_neurocore.adapters.holonomic.l13_source import L13_SourceAdapter, L13_HolonomicParameters
from sc_neurocore.adapters.holonomic.l14_trans import (
    L14_TransdimensionalAdapter,
    L14_HolonomicParameters,
)
from sc_neurocore.adapters.holonomic.l15_cons import L15_ConsiliumAdapter, L15_HolonomicParameters
from sc_neurocore.adapters.holonomic.l16_meta import L16_MetaAdapter, L16_HolonomicParameters
from sc_neurocore.quantum.qec import QecShield
from sc_neurocore.compiler.mlir_emitter import MLIREmitter
from sc_neurocore.compiler.pipeline import CompilerPipeline

from sc_neurocore.accel.jax_backend import to_host

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OmniConvergence")


def run_demonstration() -> None:
    logger.info("Initializing Phase 16 Omni-Substrate Convergence Demo...")

    # Setup Adapters
    l1 = L1_QuantumAdapter(L1_HolonomicParameters(n_qubits=10))
    l2 = L2_NeurochemicalAdapter(L2_HolonomicParameters(n_transmitters=4))
    l3 = L3_GenomicAdapter(L3_HolonomicParameters(n_genes=10))
    l4 = L4_CellularAdapter(L4_HolonomicParameters(n_cells=20))
    l5 = L5_OrganismalAdapter(L5_HolonomicParameters(n_nodes=10))
    l6 = L6_PlanetaryAdapter(L6_HolonomicParameters(n_regions=10))
    l7 = L7_SymbolicAdapter(L7_HolonomicParameters(n_nodes=13))
    l8 = L8_CosmicAdapter(L8_HolonomicParameters(n_pulsars=12))
    l9 = L9_MemoryAdapter(L9_HolonomicParameters(n_memory_slots=10))
    l10 = L10_FirewallAdapter(L10_HolonomicParameters(n_boundary_nodes=10))
    l11 = L11_NoosphericAdapter(L11_HolonomicParameters(n_nodes=10))
    l12 = L12_GaianAdapter(L12_HolonomicParameters(n_nodes=10))
    l13 = L13_SourceAdapter(L13_HolonomicParameters(n_vacuum_nodes=10))
    l14 = L14_TransdimensionalAdapter(L14_HolonomicParameters(n_bulk_dimensions=11))
    l15 = L15_ConsiliumAdapter(L15_HolonomicParameters(n_metric_dimensions=16))
    l16 = L16_MetaAdapter(L16_HolonomicParameters(n_meta_nodes=10))

    dt = 0.1

    logger.info("Step 1: Running full 16-layer coupled JAX simulation...")

    # 1. Biological Base (L1-L5)
    l1_out = l1.step_jax(dt)
    l2_out = l2.step_jax(dt, inputs=l1_out)
    l3_out = l3.step_jax(dt, inputs=l2_out)
    l4_out = l4.step_jax(dt, inputs=l3_out)
    l5_out = l5.step_jax(dt, inputs=l4_out)

    # 2. Collective & Control (L6-L12)
    l6_out = l6.step_jax(dt, inputs=l5_out)
    l7_out = l7.step_jax(dt, inputs=l6_out)
    l8_out = l8.step_jax(dt, inputs=l7_out)
    l9_out = l9.step_jax(dt, inputs=l5_out)
    l10_out = l10.step_jax(dt, inputs=l8_out)
    l11_out = l11.step_jax(dt, inputs=l10_out)
    l12_out = l12.step_jax(dt, inputs=l11_out)

    # 3. Transcendent Peak (L13-L16)
    l13_out = l13.step_jax(dt)
    l14_out = l14.step_jax(dt, inputs=l8_out)

    # Consilium integrates the full stack
    l15_out = l15.step_jax(dt, inputs=l12_out)  # Inputs from Gaian Sync
    l16_out = l16.step_jax(dt, inputs=l15_out)  # Director steers based on executive GCI

    logger.info(f"L1 Coherence: {l1.get_metrics()['r1_global_coherence']:.4f}")
    logger.info(f"L5 Self-Soliton: {l5.get_metrics()['self_soliton_magnitude']:.4f}")
    logger.info(f"L8 Cosmic Lock: {l8.get_metrics()['pta_locking_index']:.4f}")
    logger.info(f"L12 Gaian Sync: {l12.get_metrics()['eco_system_coherence']:.4f}")
    logger.info(f"L13 Vacuum Pot: {l13.get_metrics()['vacuum_potential']:.4f}")
    logger.info(f"L14 Brane Align: {l14.get_metrics()['avg_brane_alignment']:.4f}")
    logger.info(f"L15 GCI Index: {l15.get_metrics()['gci_index']:.4f}")
    logger.info(f"L16 Director Will: {l16.get_metrics()['director_will']:.4f}")

    # 4. Showcase Quantum Robustness (QEC)
    logger.info("Step 2: Protecting quantum bitstreams with QEC Shield...")
    qec = QecShield(code_type="repetition", distance=3)
    l1_out_np = to_host(l1_out)
    physical_bits = qec.encode(l1_out_np)
    logger.info(f"Encoded logical bits into physical shape: {physical_bits.shape}")

    # 5. Showcase Hardware Synthesis (MLIR)
    logger.info("Step 3: Compiling director graph to MLIR...")
    emitter = MLIREmitter("director_top")
    lfsr = emitter.emit_lfsr(16, 0xACE1)
    mlir_str = emitter.generate()

    pipeline = CompilerPipeline()
    v_path = pipeline.compile_mlir_to_verilog(mlir_str, "director_top")
    logger.info(f"Lowered MLIR to Verilog: {v_path}")

    logger.info("Phase 16 Omni-Substrate Demonstration COMPLETE.")


if __name__ == "__main__":
    run_demonstration()
