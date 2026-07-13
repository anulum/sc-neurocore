# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.bioware -- Biological-hardware interface (organoids, MEA)

"""sc_neurocore.bioware -- Biological-hardware interface (organoids, MEA).

Tier: experimental.
"""

__tier__ = "experimental"

from .bioware import (
    AEREvent,
    AERToSCConverter,
    ArtifactRejector,
    BCMPlasticity,
    BioAuditEntry,
    BioAuditLog,
    BioHybridFrameResult,
    BioHybridSession,
    BiologicalSTDP,
    CultureHealth,
    DEFAULT_LFP_BANDS,
    DetectedSpike,
    HomeostaticPlasticity,
    LFPBand,
    LatencyBudget,
    MEAConfig,
    MEALayout,
    MEAToAERTranscoder,
    MultiWellPlate,
    NetworkBurst,
    OptogeneticPulse,
    PharmModel,
    SCToOptoEncoder,
    SpikeDetector,
    SpikeSorter,
    StimProtocol,
    WellConfig,
    decode_bitstream_rate,
    detect_network_bursts,
    extract_lfp_power,
    mea_fitness_hook,
)

__all__ = [
    "AEREvent",
    "AERToSCConverter",
    "ArtifactRejector",
    "BCMPlasticity",
    "BioAuditEntry",
    "BioAuditLog",
    "BioHybridFrameResult",
    "BioHybridSession",
    "BiologicalSTDP",
    "CultureHealth",
    "DEFAULT_LFP_BANDS",
    "DetectedSpike",
    "HomeostaticPlasticity",
    "LFPBand",
    "LatencyBudget",
    "MEAConfig",
    "MEALayout",
    "MEAToAERTranscoder",
    "MultiWellPlate",
    "NetworkBurst",
    "OptogeneticPulse",
    "PharmModel",
    "SCToOptoEncoder",
    "SpikeDetector",
    "SpikeSorter",
    "StimProtocol",
    "WellConfig",
    "decode_bitstream_rate",
    "detect_network_bursts",
    "extract_lfp_power",
    "mea_fitness_hook",
]
