# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Historical bioware compatibility facade

"""Historical import facade for the biological-hardware interface.

Implementations live in responsibility-specific sibling modules. This module
retains the established import, qualified-name, object-identity, and pickle
contracts for callers of :mod:`sc_neurocore.bioware.bioware`.
"""

from .bioware_acquisition import ArtifactRejector, SpikeDetector, SpikeSorter
from .bioware_analysis import (
    DEFAULT_LFP_BANDS,
    CultureHealth,
    LFPBand,
    LatencyBudget,
    NetworkBurst,
    detect_network_bursts,
    extract_lfp_power,
)
from .bioware_audit import BioAuditEntry, BioAuditLog
from .bioware_contracts import (
    AEREvent,
    BioHybridFrameResult,
    DetectedSpike,
    MEAConfig,
    MEALayout,
    OptogeneticPulse,
    StimProtocol,
)
from .bioware_encoding import (
    AERToSCConverter,
    MEAToAERTranscoder,
    SCToOptoEncoder,
    decode_bitstream_rate,
)
from .bioware_experiment import (
    MultiWellPlate,
    PharmModel,
    WellConfig,
    _clone_spike as _clone_spike,
    _quantile_indices as _quantile_indices,
)
from .bioware_fitness import (
    _mea_response_latency_ms as _mea_response_latency_ms,
    mea_fitness_hook,
)
from .bioware_plasticity import BCMPlasticity, BiologicalSTDP, HomeostaticPlasticity
from .bioware_session import BioHybridSession

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

_COMPATIBILITY_SYMBOLS = (
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
    _clone_spike,
    _mea_response_latency_ms,
    _quantile_indices,
    decode_bitstream_rate,
    detect_network_bursts,
    extract_lfp_power,
    mea_fitness_hook,
)

for _symbol in _COMPATIBILITY_SYMBOLS:
    _symbol.__module__ = __name__

del _symbol
