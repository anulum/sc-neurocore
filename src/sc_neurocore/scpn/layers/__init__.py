from typing import Any, Optional
"""
SCPN Layers Module
==================

Stochastic implementations of the SCPN (Self-Consistent Phenomenological Network)
layer hierarchy for the sc-neurocore framework.

Layer Hierarchy:
- L1: Quantum Biological (microtubules, NV centers)
- L2: Neurochemical (receptors, neurotransmitters)
- L3: Genomic-Epigenomic (CISS, bioelectric, chromatin)
- L4: Cellular-Tissue Synchronization (gap junctions, calcium waves)
- L5: Organismal-Psychoemotional (HRV, autonomic, emotions)
- L6: Ecological-Planetary (Schumann, geomagnetic, biosphere)
- L7: Geometric-Symbolic (sacred geometry, E8, acupuncture)

"""

from .l1_quantum import L1_QuantumLayer, L1_StochasticParameters
from .l2_neurochemical import L2_NeurochemicalLayer, L2_StochasticParameters
from .l3_genomic import L3_GenomicLayer, L3_StochasticParameters
from .l4_cellular import L4_CellularLayer, L4_StochasticParameters
from .l5_organismal import L5_OrganismalLayer, L5_StochasticParameters
from .l6_ecological import L6_EcologicalLayer, L6_StochasticParameters
from .l7_symbolic import L7_SymbolicLayer, L7_StochasticParameters
from typing import Optional

__all__ = [
    # L1 Quantum
    "L1_QuantumLayer",
    "L1_StochasticParameters",
    # L2 Neurochemical
    "L2_NeurochemicalLayer",
    "L2_StochasticParameters",
    # L3 Genomic
    "L3_GenomicLayer",
    "L3_StochasticParameters",
    # L4 Cellular
    "L4_CellularLayer",
    "L4_StochasticParameters",
    # L5 Organismal
    "L5_OrganismalLayer",
    "L5_StochasticParameters",
    # L6 Ecological
    "L6_EcologicalLayer",
    "L6_StochasticParameters",
    # L7 Symbolic
    "L7_SymbolicLayer",
    "L7_StochasticParameters",
]


def create_full_stack(params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """
    Create a complete SCPN layer stack with default or custom parameters.

    Args:
        params: Optional dict with layer-specific parameter overrides.
                Keys: 'l1', 'l2', 'l3', 'l4', 'l5', 'l6', 'l7'

    Returns:
        Dict with layer instances keyed by layer name.
    """
    params = params or {}

    return {
        "l1": L1_QuantumLayer(params.get("l1")),
        "l2": L2_NeurochemicalLayer(params.get("l2")),
        "l3": L3_GenomicLayer(params.get("l3")),
        "l4": L4_CellularLayer(params.get("l4")),
        "l5": L5_OrganismalLayer(params.get("l5")),
        "l6": L6_EcologicalLayer(params.get("l6")),
        "l7": L7_SymbolicLayer(params.get("l7")),
    }


def run_integrated_step(layers: dict[str, Any], dt: float, inputs: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """
    Run one integrated time step across all SCPN layers with inter-layer coupling.

    Args:
        layers: Dict of layer instances from create_full_stack().
        dt: Time step in seconds.
        inputs: Optional external inputs for specific layers.

    Returns:
        Dict with outputs from each layer.
    """
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

    return outputs


def get_global_metrics(layers: dict[str, Any]) -> dict[str, Any]:
    """Get global coherence metrics from all layers."""
    return {
        "l1_quantum_coherence": layers["l1"].get_global_metric(),
        "l2_neurochemical_activity": layers["l2"].get_global_metric(),
        "l3_genomic_expression": layers["l3"].get_global_metric(),
        "l4_cellular_sync": layers["l4"].get_global_metric(),
        "l5_organismal_coherence": layers["l5"].get_global_metric(),
        "l6_planetary_coherence": layers["l6"].get_global_metric(),
        "l7_symbolic_health": layers["l7"].get_global_metric(),
    }
