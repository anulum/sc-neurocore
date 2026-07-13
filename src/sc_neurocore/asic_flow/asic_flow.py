# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Historical ASIC-flow compatibility facade

"""Preserve the original ASIC-flow API over responsibility modules.

New code may import the focused modules directly. Historical imports and
pickle-qualified names remain stable through this facade.
"""

from __future__ import annotations

from sc_neurocore.asic_flow.constraints import (
    CDCCheckGenerator,
    IOConstraintGenerator,
    IOPin,
    IRDropGenerator,
    LECGenerator,
)
from sc_neurocore.asic_flow.decks import (
    FloorplanGenerator,
    GDSIIExporter,
    PlaceRouteGenerator,
    SDCGenerator,
    SynthesisGenerator,
)
from sc_neurocore.asic_flow.design import DesignParams, SCASICOptimisationConfig
from sc_neurocore.asic_flow.estimation import DesignEstimate, PreSynthEstimator
from sc_neurocore.asic_flow.flow import (
    ASICFlowBundle,
    ASICFlowGenerator,
    ASICFlowOutput,
    _build_asic_flow_manifest as _build_asic_flow_manifest,
    _design_to_manifest as _design_to_manifest,
    _formal_evidence_status as _formal_evidence_status,
    _normalise_pdk_type as _normalise_pdk_type,
    _pdk_to_manifest as _pdk_to_manifest,
    generate_asic_flow_bundle,
)
from sc_neurocore.asic_flow.hierarchy import BlockConfig, HierarchicalFlow
from sc_neurocore.asic_flow.pdk import (
    OpenSourcePDKResolver,
    PDKConfig,
    PDKResolution,
    PDKType,
    PDKValidationResult,
    ResolvedPDKFiles,
    validate_pdk,
    validate_pdk_installation,
)
from sc_neurocore.asic_flow.readiness import TapeOutChecklist
from sc_neurocore.asic_flow.signoff import (
    DEFAULT_CORNERS,
    CornerType,
    DRCViolation,
    MultiCornerAnalysis,
    OCVConfig,
    PVTCorner,
    SignoffCheckResult,
    SignoffGenerator,
    SignoffSummary,
)

__all__ = [
    "ASICFlowBundle",
    "ASICFlowGenerator",
    "ASICFlowOutput",
    "BlockConfig",
    "CDCCheckGenerator",
    "CornerType",
    "DEFAULT_CORNERS",
    "DRCViolation",
    "DesignEstimate",
    "DesignParams",
    "FloorplanGenerator",
    "GDSIIExporter",
    "HierarchicalFlow",
    "IOConstraintGenerator",
    "IOPin",
    "IRDropGenerator",
    "LECGenerator",
    "MultiCornerAnalysis",
    "OCVConfig",
    "OpenSourcePDKResolver",
    "PDKConfig",
    "PDKResolution",
    "PDKType",
    "PDKValidationResult",
    "PVTCorner",
    "PlaceRouteGenerator",
    "PreSynthEstimator",
    "ResolvedPDKFiles",
    "SCASICOptimisationConfig",
    "SDCGenerator",
    "SignoffCheckResult",
    "SignoffGenerator",
    "SignoffSummary",
    "SynthesisGenerator",
    "TapeOutChecklist",
    "generate_asic_flow_bundle",
    "validate_pdk",
    "validate_pdk_installation",
]

_HISTORICAL_DEFINITIONS = [
    *__all__[:6],
    *__all__[7:],
    "_build_asic_flow_manifest",
    "_design_to_manifest",
    "_formal_evidence_status",
    "_normalise_pdk_type",
    "_pdk_to_manifest",
]

for _historical_name in _HISTORICAL_DEFINITIONS:
    globals()[_historical_name].__module__ = __name__

del _historical_name
del _HISTORICAL_DEFINITIONS
