# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_holonomic_adapter_stack_contract.py

from __future__ import annotations

"""Workflow contract tests for the L1-L16 holonomic adapter stack."""


import subprocess


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


from sc_neurocore.adapters.holonomic.l13_source import L13_SourceAdapter, L13_HolonomicParameters


from sc_neurocore.adapters.holonomic.l14_trans import (
    L14_TransdimensionalAdapter,
    L14_HolonomicParameters,
)


from sc_neurocore.adapters.holonomic.l15_cons import L15_ConsiliumAdapter, L15_HolonomicParameters


from sc_neurocore.adapters.holonomic.l16_meta import L16_MetaAdapter, L16_HolonomicParameters


from sc_neurocore.adapters.base import BaseStochasticAdapter


from sc_neurocore.audio.user_profile import UserProfile, Chronotype


__all__ = [
    "subprocess",
    "pytest",
    "np",
    "jnp",
    "L1_QuantumAdapter",
    "L1_HolonomicParameters",
    "L2_NeurochemicalAdapter",
    "L2_HolonomicParameters",
    "L3_GenomicAdapter",
    "L3_HolonomicParameters",
    "L4_CellularAdapter",
    "L5_OrganismalAdapter",
    "L6_HolonomicParameters",
    "L6_PlanetaryAdapter",
    "L11_NoosphericAdapter",
    "L12_GaianAdapter",
    "QecShield",
    "CompilerPipeline",
    "L7_HolonomicParameters",
    "L7_SymbolicAdapter",
    "L8_CosmicAdapter",
    "L9_MemoryAdapter",
    "L10_FirewallAdapter",
    "DNAEncoder",
    "GeneticRegulatoryLayer",
    "NeuromodulatorSystem",
    "L13_SourceAdapter",
    "L13_HolonomicParameters",
    "L14_TransdimensionalAdapter",
    "L14_HolonomicParameters",
    "L15_ConsiliumAdapter",
    "L15_HolonomicParameters",
    "L16_MetaAdapter",
    "L16_HolonomicParameters",
    "BaseStochasticAdapter",
    "UserProfile",
    "Chronotype",
]
