# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Automated Safety & Regulatory Certification Generator

"""Compatibility façade for the modular safety-evidence implementation.

New code may import responsibility modules directly. Existing imports and
pickle module paths remain stable through this façade.
"""

from __future__ import annotations

from sc_neurocore.safety_cert.certification import (
    CertificationGenerator,
    CertificationPackage,
    SafetyManualGenerator,
)
from sc_neurocore.safety_cert.change_impact import ChangeImpactTracker, ChangeRecord
from sc_neurocore.safety_cert.compliance import (
    CROSS_MAP as CROSS_MAP,
    ChecklistItem,
    ComplianceChecklist,
    CrossStandardMapper,
    IEC62304Assessment,
    SWClass,
)
from sc_neurocore.safety_cert.evidence import EvidenceBag, EvidenceItem
from sc_neurocore.safety_cert.failure_analysis import (
    FMEDA,
    FailureCategory,
    FailureMode,
    ReliabilityMetrics,
)
from sc_neurocore.safety_cert.fault_tolerance import (
    CCFAnalysis,
    CCFDefence,
    HFTAssessment,
    HFTLevel,
)
from sc_neurocore.safety_cert.formal_evidence import (
    FormalProofCertificate,
    FormalProperty,
    FormalPropertyGapDetector,
    ProofTestCoverage,
    PropertyGap,
)
from sc_neurocore.safety_cert.standards import (
    ASILLevel,
    SIL_TO_ASIL as SIL_TO_ASIL,
    SILLevel,
    SafetyStandard,
)
from sc_neurocore.safety_cert.timing_analysis import WCETAnalyzer, WCETPath
from sc_neurocore.safety_cert.traceability import Requirement, TraceabilityMatrix

__all__ = [
    "ASILLevel",
    "CCFAnalysis",
    "CCFDefence",
    "CertificationGenerator",
    "CertificationPackage",
    "ChangeImpactTracker",
    "ChangeRecord",
    "ChecklistItem",
    "ComplianceChecklist",
    "CrossStandardMapper",
    "EvidenceBag",
    "EvidenceItem",
    "FMEDA",
    "FailureCategory",
    "FailureMode",
    "FormalProofCertificate",
    "FormalProperty",
    "FormalPropertyGapDetector",
    "HFTAssessment",
    "HFTLevel",
    "IEC62304Assessment",
    "ProofTestCoverage",
    "PropertyGap",
    "ReliabilityMetrics",
    "Requirement",
    "SafetyManualGenerator",
    "SafetyStandard",
    "SILLevel",
    "SWClass",
    "TraceabilityMatrix",
    "WCETAnalyzer",
    "WCETPath",
]

# Historical pickles resolve these public classes through this compatibility module.
for _public_name in __all__:
    globals()[_public_name].__module__ = __name__
del _public_name
