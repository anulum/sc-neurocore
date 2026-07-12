# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Speculative hardware bridges for non-von-Neumann substrates

"""Speculative hardware bridges for non-von-Neumann substrates.

This package provides compilers and mappers that translate SC bitstream
networks onto radically different physical computing substrates:

- **dna_mapper** — Molecular/DNA strand displacement circuits
- **quantum_annealing** — D-Wave / Ising model export
- **photonic_noc** — Neuromorphic optical interconnect fabric

Each bridge follows the same pattern:

1. Accept an SC network or raw bitstream as input.
2. Compile to the target substrate's native representation.
3. Simulate the compiled design (classical fallback available).
4. Export to the substrate's toolchain format.

All external dependencies (NUPACK, D-Wave Ocean, gdstk) use soft imports
with graceful fallback to internal simulation engines.
"""

__tier__ = "research"

from .dna_mapper import (
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

from .quantum_annealing import (
    AnnealingSchedule,
    ChainBreakResolver,
    CouplerSpec,
    DWaveInterface,
    EmbeddingAnalyzer,
    EnergyLandscape,
    GaugeTransform,
    HardwareGraph,
    IsingModel,
    ProblemDecomposer,
    ProblemType,
    QUBOModel,
    QubitSpec,
    SampleAggregator,
    SCBitstreamQUBO,
    SCPrecisionEncoder,
    SCToIsing,
    SCToQUBO,
    SimulatedAnnealer,
    TTSAnalyzer,
    export_bqm,
    export_ising_json,
    export_qubo_json,
    visualize_ising,
)

__all__ += [
    "AnnealingSchedule",
    "ChainBreakResolver",
    "CouplerSpec",
    "DWaveInterface",
    "EmbeddingAnalyzer",
    "EnergyLandscape",
    "GaugeTransform",
    "HardwareGraph",
    "IsingModel",
    "ProblemDecomposer",
    "ProblemType",
    "QUBOModel",
    "QubitSpec",
    "SampleAggregator",
    "SCBitstreamQUBO",
    "SCPrecisionEncoder",
    "SCToIsing",
    "SCToQUBO",
    "SimulatedAnnealer",
    "TTSAnalyzer",
    "export_bqm",
    "export_ising_json",
    "export_qubo_json",
    "visualize_ising",
]

from .photonic_noc import (
    CrosstalkAnalyzer,
    MZICompiler,
    MZIGate,
    PhotonicCircuitDesign,
    PowerBudgetAnalyzer,
    SCToPhotonic,
    ThermalPhaseShifter,
    WDMAssigner,
    WDMChannel,
    WaveguideRouter,
    WaveguideSegment,
    WaveguideType,
    export_photonic_json,
    visualize_photonic,
)

__all__ += [
    "CrosstalkAnalyzer",
    "MZICompiler",
    "MZIGate",
    "PhotonicCircuitDesign",
    "PowerBudgetAnalyzer",
    "SCToPhotonic",
    "ThermalPhaseShifter",
    "WDMAssigner",
    "WDMChannel",
    "WaveguideRouter",
    "WaveguideSegment",
    "WaveguideType",
    "export_photonic_json",
    "visualize_photonic",
]

try:
    from .photonic_codesign import (
        BitstreamEvidence,
        PhotonicCoDesignConfig,
        PhotonicCoDesignReport,
        StochasticPhotonicCoDesignLoop,
        derive_probabilities_from_adjacency,
        encode_bitstream_bank,
    )
except ModuleNotFoundError as exc:
    if exc.name != "sc_neurocore.optics":
        raise
else:
    __all__ += [
        "BitstreamEvidence",
        "PhotonicCoDesignConfig",
        "PhotonicCoDesignReport",
        "StochasticPhotonicCoDesignLoop",
        "derive_probabilities_from_adjacency",
        "encode_bitstream_bank",
    ]

from .local_llm import (
    LocalLLMBridge,
    LocalLLMConfig,
    LocalLLMError,
    LocalLLMProvider,
    LocalLLMResponse,
    SpikePromptAdapter,
)

__all__ += [
    "LocalLLMBridge",
    "LocalLLMConfig",
    "LocalLLMError",
    "LocalLLMProvider",
    "LocalLLMResponse",
    "SpikePromptAdapter",
]
