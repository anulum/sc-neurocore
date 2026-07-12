# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Molecular/DNA Computing Mapper

"""Compatibility façade for the molecular/DNA computing mapper.

The responsibility modules behind this façade compile stochastic-computing
Boolean networks into strand-displacement or enzymatic circuits, simulate the
resulting kinetics, validate thermodynamics, and export laboratory artefacts.
Historical imports from :mod:`sc_neurocore.bridges.dna_mapper` remain stable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from sc_neurocore_engine.dna import has_full_dna_backend
except ImportError:

    def has_full_dna_backend() -> bool:
        """Return whether the optional Rust DNA backend is available."""
        return False


try:
    import nupack as nupack

    _HAS_NUPACK = True
except ImportError:
    nupack = None
    _HAS_NUPACK = False

try:
    _HAS_RUST_DNA = has_full_dna_backend()
except ImportError:
    _HAS_RUST_DNA = False

from .dna_analysis import (
    CrossHybridizationChecker,
    GateOptimizer,
    HairpinChecker,
    TopologicalAnalyzer,
)
from .dna_bridge import BitstreamToDNA, SCNetworkBridge
from .dna_compilers import EnzymaticGateCompiler, StrandDisplacementCompiler
from .dna_encoding import DualRailEncoder, GF4ErrorCorrection
from .dna_io import (
    PlateLayout,
    estimate_cost,
    export_fasta,
    export_genbank,
    export_json,
    export_nupack_input,
    generate_protocol,
    visualize_circuit,
    visualize_kinetics,
)
from .dna_sequences import SequenceDesigner
from .dna_simulation import (
    ConcentrationOptimizer,
    DegradationModel,
    KineticSimulator,
    NoiseModel,
    SCPrecisionAnalyzer,
)
from .dna_thermodynamics import (
    NUPACKInterface,
    _can_pair,
    _canonical_sequence,
    _configure_nupack_backend,
    _fallback_pair_energy,
    _fallback_pair_probability_matrix,
    _fallback_secondary_structure,
    _hairpin_loop_penalty,
)

_configure_nupack_backend(lambda: (_HAS_NUPACK, nupack))

from .dna_types import (
    _CLAMP_LENGTH,
    _DEFAULT_TEMPERATURE_C,
    _GC_TARGET_HIGH,
    _GC_TARGET_LOW,
    _HAIRPIN_LOOP_INIT_DG,
    _HAIRPIN_LOOP_SLOPE_DG,
    _MAX_HOMOPOLYMER,
    _MIN_HAIRPIN_LOOP_NT,
    _NN_DG,
    _NN_DH,
    _NN_DS,
    _NN_INIT_DG,
    _NN_INIT_DH,
    _NN_INIT_DS,
    _R_GAS,
    _RECOGNITION_LENGTH,
    _STACKING_BONUS_DG,
    _TOEHOLD_LENGTH,
    _WC_PAIR_DG,
    CompilationMethod,
    DNACircuitDesign,
    DNAGate,
    DNAStrand,
    GateType,
)

__all__ = [
    "BitstreamToDNA",
    "CompilationMethod",
    "ConcentrationOptimizer",
    "CrossHybridizationChecker",
    "DNACircuitDesign",
    "DNAGate",
    "DNAStrand",
    "DegradationModel",
    "DualRailEncoder",
    "EnzymaticGateCompiler",
    "GF4ErrorCorrection",
    "GateOptimizer",
    "GateType",
    "HairpinChecker",
    "KineticSimulator",
    "NUPACKInterface",
    "NoiseModel",
    "PlateLayout",
    "SCNetworkBridge",
    "SCPrecisionAnalyzer",
    "SequenceDesigner",
    "StrandDisplacementCompiler",
    "TopologicalAnalyzer",
    "estimate_cost",
    "export_fasta",
    "export_genbank",
    "export_json",
    "export_nupack_input",
    "generate_protocol",
    "visualize_circuit",
    "visualize_kinetics",
]

_COMPATIBILITY_VALUES: tuple[Any, ...] = (
    np,
    _CLAMP_LENGTH,
    _DEFAULT_TEMPERATURE_C,
    _GC_TARGET_HIGH,
    _GC_TARGET_LOW,
    _HAIRPIN_LOOP_INIT_DG,
    _HAIRPIN_LOOP_SLOPE_DG,
    _MAX_HOMOPOLYMER,
    _MIN_HAIRPIN_LOOP_NT,
    _NN_DG,
    _NN_DH,
    _NN_DS,
    _NN_INIT_DG,
    _NN_INIT_DH,
    _NN_INIT_DS,
    _R_GAS,
    _RECOGNITION_LENGTH,
    _STACKING_BONUS_DG,
    _TOEHOLD_LENGTH,
    _WC_PAIR_DG,
    _can_pair,
    _canonical_sequence,
    _fallback_pair_energy,
    _fallback_pair_probability_matrix,
    _fallback_secondary_structure,
    _hairpin_loop_penalty,
)

_COMPATIBILITY_OBJECTS: tuple[Any, ...] = (
    BitstreamToDNA,
    CompilationMethod,
    ConcentrationOptimizer,
    CrossHybridizationChecker,
    DNACircuitDesign,
    DNAGate,
    DNAStrand,
    DegradationModel,
    DualRailEncoder,
    EnzymaticGateCompiler,
    GF4ErrorCorrection,
    GateOptimizer,
    GateType,
    HairpinChecker,
    KineticSimulator,
    NUPACKInterface,
    NoiseModel,
    PlateLayout,
    SCNetworkBridge,
    SCPrecisionAnalyzer,
    SequenceDesigner,
    StrandDisplacementCompiler,
    TopologicalAnalyzer,
    estimate_cost,
    export_fasta,
    export_genbank,
    export_json,
    export_nupack_input,
    generate_protocol,
    visualize_circuit,
    visualize_kinetics,
)

for _compatibility_object in _COMPATIBILITY_OBJECTS:
    _compatibility_object.__module__ = __name__

del _compatibility_object
