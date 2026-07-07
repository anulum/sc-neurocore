# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN Layers Module — Full 16-Layer Hierarchy

"""Expose stochastic implementations for the full 16-layer SCPN hierarchy.

Stochastic (pure NumPy) implementations of all 16 SCPN layers.

L1-L7:  Biological/physical substrate (quantum → symbolic)
L8-L10: Cosmic/memory/boundary control
L11-L13: Informational/ecological/temporal binding
L14-L16: Integration, meta-cognition, cybernetic closure
"""

from typing import Any, Optional

from .l1_quantum import L1_QuantumLayer, L1_StochasticParameters
from .l2_neurochemical import L2_NeurochemicalLayer, L2_StochasticParameters
from .l3_genomic import L3_GenomicLayer, L3_StochasticParameters
from .l4_cellular import L4_CellularLayer, L4_StochasticParameters
from .l5_organismal import L5_OrganismalLayer, L5_StochasticParameters
from .l6_ecological import L6_EcologicalLayer, L6_StochasticParameters
from .l7_symbolic import L7_SymbolicLayer, L7_StochasticParameters
from .l8_phase_field import L8_PhaseFieldLayer, L8_StochasticParameters
from .l9_memory import L9_MemoryLayer, L9_StochasticParameters
from .l10_boundary import L10_BoundaryLayer, L10_StochasticParameters
from .l11_morphic import L11_MorphicLayer, L11_StochasticParameters
from .l12_quantum_info import L12_QuantumInfoLayer, L12_StochasticParameters
from .l13_temporal import L13_TemporalLayer, L13_StochasticParameters
from .l14_integration import L14_IntegrationLayer, L14_StochasticParameters
from .l15_meta import L15_MetaLayer, L15_StochasticParameters
from .l16_director import L16_DirectorLayer, L16_StochasticParameters

__all__ = [
    "L1_QuantumLayer",
    "L1_StochasticParameters",
    "L2_NeurochemicalLayer",
    "L2_StochasticParameters",
    "L3_GenomicLayer",
    "L3_StochasticParameters",
    "L4_CellularLayer",
    "L4_StochasticParameters",
    "L5_OrganismalLayer",
    "L5_StochasticParameters",
    "L6_EcologicalLayer",
    "L6_StochasticParameters",
    "L7_SymbolicLayer",
    "L7_StochasticParameters",
    "L8_PhaseFieldLayer",
    "L8_StochasticParameters",
    "L9_MemoryLayer",
    "L9_StochasticParameters",
    "L10_BoundaryLayer",
    "L10_StochasticParameters",
    "L11_MorphicLayer",
    "L11_StochasticParameters",
    "L12_QuantumInfoLayer",
    "L12_StochasticParameters",
    "L13_TemporalLayer",
    "L13_StochasticParameters",
    "L14_IntegrationLayer",
    "L14_StochasticParameters",
    "L15_MetaLayer",
    "L15_StochasticParameters",
    "L16_DirectorLayer",
    "L16_StochasticParameters",
    "LAYER_REGISTRY",
    "create_full_stack",
    "run_integrated_step",
    "get_global_metrics",
]

LAYER_REGISTRY = {
    "l1": L1_QuantumLayer,
    "l2": L2_NeurochemicalLayer,
    "l3": L3_GenomicLayer,
    "l4": L4_CellularLayer,
    "l5": L5_OrganismalLayer,
    "l6": L6_EcologicalLayer,
    "l7": L7_SymbolicLayer,
    "l8": L8_PhaseFieldLayer,
    "l9": L9_MemoryLayer,
    "l10": L10_BoundaryLayer,
    "l11": L11_MorphicLayer,
    "l12": L12_QuantumInfoLayer,
    "l13": L13_TemporalLayer,
    "l14": L14_IntegrationLayer,
    "l15": L15_MetaLayer,
    "l16": L16_DirectorLayer,
}


def create_full_stack(params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Create a complete 16-layer SCPN stack."""
    params = params or {}
    return {key: cls(params.get(key)) for key, cls in LAYER_REGISTRY.items()}


def run_integrated_step(
    layers: dict[str, Any], dt: float, inputs: Optional[dict[str, Any]] = None
) -> dict[str, Any]:
    """Run one integrated time step across all SCPN layers with inter-layer coupling."""
    inputs = inputs or {}
    outputs = {}

    # L1: Quantum (foundation)
    l1_bitstreams = layers["l1"].step(dt, external_field=inputs.get("l1_field"))
    outputs["l1"] = {
        "output_bitstreams": l1_bitstreams,
        "coherence": layers["l1"].get_global_metric(),
    }

    # L2: Neurochemical (receives L1 quantum modulation)
    l2_out = layers["l2"].step(dt, nt_release=inputs.get("nt_release"), l1_input=l1_bitstreams)
    outputs["l2"] = l2_out

    # L3: Genomic (receives L2 second messengers)
    l3_out = layers["l3"].step(dt, l2_input=l2_out, bioelectric_signal=inputs.get("bioelectric"))
    outputs["l3"] = l3_out

    # L4: Cellular (receives L3 protein modulation)
    l4_out = layers["l4"].step(
        dt, l3_input=l3_out, external_stimulus=inputs.get("cellular_stimulus")
    )
    outputs["l4"] = l4_out

    # L5: Organismal (receives L4 synchronization)
    l5_out = layers["l5"].step(dt, l4_input=l4_out, external_event=inputs.get("emotional_event"))
    outputs["l5"] = l5_out

    # L6: Ecological (receives L5 organismal state)
    l6_out = layers["l6"].step(
        dt,
        l5_input=l5_out,
        solar_activity=inputs.get("solar", 0.5),
        lunar_phase=inputs.get("lunar", 0.0),
    )
    outputs["l6"] = l6_out

    # L7: Symbolic (receives L6 Schumann/ecological)
    l7_out = layers["l7"].step(
        dt,
        l6_input=l6_out,
        symbol_input=inputs.get("symbols"),
        acupoint_stimulus=inputs.get("acupoints"),
    )
    outputs["l7"] = l7_out

    # L8: Phase Field / Cosmic (receives L7 symbolic)
    l8_out = layers["l8"].step(dt, l7_input=l7_out)
    outputs["l8"] = l8_out

    # L9: Memory (receives L8 cosmic alignment)
    l9_out = layers["l9"].step(dt, l8_input=l8_out)
    outputs["l9"] = l9_out

    # L10: Boundary (receives L9 retrieval quality)
    l10_out = layers["l10"].step(dt, l9_input=l9_out)
    outputs["l10"] = l10_out

    # L11: Morphic (receives L10 integrity)
    l11_out = layers["l11"].step(dt, l10_input=l10_out)
    outputs["l11"] = l11_out

    # L12: Quantum Info (receives L11 info saturation)
    l12_out = layers["l12"].step(dt, l11_input=l11_out)
    outputs["l12"] = l12_out

    # L13: Temporal (receives L12 coherence)
    l13_out = layers["l13"].step(dt, l12_input=l12_out)
    outputs["l13"] = l13_out

    # L14: Integration (receives metrics from all layers)
    all_metrics = get_global_metrics(layers)
    l14_out = layers["l14"].step(dt, layer_metrics=all_metrics, l13_input=l13_out)
    outputs["l14"] = l14_out

    # L15: Meta-cognitive (receives L14 integrated coherence)
    l15_out = layers["l15"].step(dt, l14_input=l14_out)
    outputs["l15"] = l15_out

    # L16: Director (receives L15 GCI)
    l16_out = layers["l16"].step(dt, l15_input=l15_out)
    outputs["l16"] = l16_out

    return outputs


def get_global_metrics(layers: dict[str, Any]) -> dict[str, Any]:
    """Get global coherence metrics from all layers."""
    return {f"l{i}": layers[f"l{i}"].get_global_metric() for i in range(1, 17) if f"l{i}" in layers}
