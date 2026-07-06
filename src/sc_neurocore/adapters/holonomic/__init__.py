# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Holonomic adapters for SCPN layers 1-16

"""Holonomic adapters for SCPN layers 1-16."""

from __future__ import annotations

from typing import Any

from sc_neurocore.utils.adapter_discovery import discover_adapters
from sc_neurocore.utils.registry import registry

from .l1_quantum import L1_QuantumAdapter
from .l2_chem import L2_NeurochemicalAdapter
from .l3_gen import L3_GenomicAdapter
from .l4_cell import L4_CellularAdapter
from .l5_org import L5_OrganismalAdapter
from .l6_plan import L6_PlanetaryAdapter
from .l7_sym import L7_SymbolicAdapter
from .l8_cosm import L8_CosmicAdapter
from .l9_mem import L9_MemoryAdapter
from .l10_fire import L10_FirewallAdapter
from .l11_noos import L11_NoosphericAdapter
from .l12_gaian import L12_GaianAdapter
from .l13_source import L13_SourceAdapter
from .l14_trans import L14_TransdimensionalAdapter
from .l15_cons import L15_ConsiliumAdapter
from .l16_meta import L16_MetaAdapter

_ADAPTERS = {
    "L1_Quantum": L1_QuantumAdapter,
    "L2_Neurochemical": L2_NeurochemicalAdapter,
    "L3_Genomic": L3_GenomicAdapter,
    "L4_Cellular": L4_CellularAdapter,
    "L5_Organismal": L5_OrganismalAdapter,
    "L6_Planetary": L6_PlanetaryAdapter,
    "L7_Symbolic": L7_SymbolicAdapter,
    "L8_Cosmic": L8_CosmicAdapter,
    "L9_Memory": L9_MemoryAdapter,
    "L10_Firewall": L10_FirewallAdapter,
    "L11_Noospheric": L11_NoosphericAdapter,
    "L12_Gaian": L12_GaianAdapter,
    "L13_Source": L13_SourceAdapter,
    "L14_Transdimensional": L14_TransdimensionalAdapter,
    "L15_Consilium": L15_ConsiliumAdapter,
    "L16_Meta": L16_MetaAdapter,
}

for _name, _cls in _ADAPTERS.items():
    registry.register("adapter", _name)(_cls)

discover_adapters(include_entry_points=False)

_LAYER_MAP = {i + 1: cls for i, cls in enumerate(_ADAPTERS.values())}


def create_adapter(layer: int) -> Any:
    """Factory: create adapter by layer number (1-16)."""
    if layer not in _LAYER_MAP:
        raise ValueError(f"Layer {layer} not in 1-16")
    return _LAYER_MAP[layer]()


__all__ = [
    "L1_QuantumAdapter",
    "L2_NeurochemicalAdapter",
    "L3_GenomicAdapter",
    "L4_CellularAdapter",
    "L5_OrganismalAdapter",
    "L6_PlanetaryAdapter",
    "L7_SymbolicAdapter",
    "L8_CosmicAdapter",
    "L9_MemoryAdapter",
    "L10_FirewallAdapter",
    "L11_NoosphericAdapter",
    "L12_GaianAdapter",
    "L13_SourceAdapter",
    "L14_TransdimensionalAdapter",
    "L15_ConsiliumAdapter",
    "L16_MetaAdapter",
    "create_adapter",
]
