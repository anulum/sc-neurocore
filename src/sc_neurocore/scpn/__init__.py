"""sc_neurocore.scpn -- Tier: research (experimental / research)."""

__tier__ = "research"

from .layers import (
    L1_QuantumLayer,
    L2_NeurochemicalLayer,
    L3_GenomicLayer,
    L4_CellularLayer,
    L5_OrganismalLayer,
    L6_EcologicalLayer,
    L7_SymbolicLayer,
    create_full_stack,
    run_integrated_step,
    get_global_metrics,
)

__all__ = [
    "L1_QuantumLayer",
    "L2_NeurochemicalLayer",
    "L3_GenomicLayer",
    "L4_CellularLayer",
    "L5_OrganismalLayer",
    "L6_EcologicalLayer",
    "L7_SymbolicLayer",
    "create_full_stack",
    "run_integrated_step",
    "get_global_metrics",
]
