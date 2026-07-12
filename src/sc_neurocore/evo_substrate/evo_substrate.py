# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Historical evolutionary-substrate compatibility facade

"""Preserve the original evolutionary-substrate API over focused modules.

New code may import from ``sc_neurocore.evo_substrate`` or the focused
responsibility modules. Historical imports and pickle-qualified names remain
stable.
"""

from __future__ import annotations

from sc_neurocore.evo_substrate.deployment import TileAllocation, TileDeploymentTracker
from sc_neurocore.evo_substrate.development import ActivationFunc, CPPNEdge, CPPNGenome, CPPNNode
from sc_neurocore.evo_substrate.ecology import (
    CoevoOrganism,
    CoevoRole,
    CoevolutionArena,
    ExtinctionDetector,
    Island,
    IslandModel,
    NoveltyArchive,
)
from sc_neurocore.evo_substrate.emission import OrganismEmitter
from sc_neurocore.evo_substrate.fitness import (
    FitnessEvaluator,
    FitnessResult,
    FitnessType,
    HWFitnessCollector,
    HWFitnessReport,
)
from sc_neurocore.evo_substrate.genome import (
    Genome,
    GenomeSerializer,
    NeuronGene,
    PlasticityGene,
    TopologyGene,
)
from sc_neurocore.evo_substrate.lineage import LineageRecord, LineageTracker
from sc_neurocore.evo_substrate.organism import Organism
from sc_neurocore.evo_substrate.replication import ReplicationEngine
from sc_neurocore.evo_substrate.safety import (
    FormalSafetyGuard,
    ResourceBudget,
    RuntimeFaultCheck,
    RuntimeFaultConfig,
    SafetyBounds,
    SafetyCheckResult,
)
from sc_neurocore.evo_substrate.selection import (
    AgeRegulator,
    BloatMetrics,
    BloatPenalizer,
    HallOfFame,
    ParetoFront,
    TournamentSelector,
    compute_bloat,
    dominates,
)
from sc_neurocore.evo_substrate.speciation import (
    _HAS_RUST_EVO as _HAS_RUST_EVO,
    _ec as _ec,
    assign_species,
    genomic_distance,
    population_diversity,
    shared_fitness,
)
from sc_neurocore.evo_substrate.statistics import (
    ComplexityTracker,
    EvoStatisticsTracker,
    GenerationStats,
    GenomeDiff,
    genome_complexity,
    genome_diff,
)
from sc_neurocore.evo_substrate.variation import (
    CrossoverEngine,
    MutationConfig,
    MutationEngine,
    MutationType,
)
from sc_neurocore.fault_injection import (
    DegradationAction as DegradationAction,
    DegradationPlan as DegradationPlan,
    FaultModel as FaultModel,
    GracefulDegradationPolicy as GracefulDegradationPolicy,
)

__all__ = [
    "ActivationFunc",
    "AgeRegulator",
    "BloatMetrics",
    "BloatPenalizer",
    "CPPNEdge",
    "CPPNGenome",
    "CPPNNode",
    "CoevoOrganism",
    "CoevoRole",
    "CoevolutionArena",
    "ComplexityTracker",
    "CrossoverEngine",
    "EvoStatisticsTracker",
    "ExtinctionDetector",
    "FitnessEvaluator",
    "FitnessResult",
    "FitnessType",
    "FormalSafetyGuard",
    "GenerationStats",
    "Genome",
    "GenomeDiff",
    "GenomeSerializer",
    "HWFitnessCollector",
    "HWFitnessReport",
    "HallOfFame",
    "Island",
    "IslandModel",
    "LineageRecord",
    "LineageTracker",
    "MutationConfig",
    "MutationEngine",
    "MutationType",
    "NeuronGene",
    "NoveltyArchive",
    "Organism",
    "OrganismEmitter",
    "ParetoFront",
    "PlasticityGene",
    "ReplicationEngine",
    "ResourceBudget",
    "RuntimeFaultCheck",
    "RuntimeFaultConfig",
    "SafetyBounds",
    "SafetyCheckResult",
    "TileAllocation",
    "TileDeploymentTracker",
    "TopologyGene",
    "TournamentSelector",
    "assign_species",
    "compute_bloat",
    "dominates",
    "genome_complexity",
    "genome_diff",
    "genomic_distance",
    "population_diversity",
    "shared_fitness",
]

for _public_name in __all__:
    globals()[_public_name].__module__ = __name__

del _public_name
