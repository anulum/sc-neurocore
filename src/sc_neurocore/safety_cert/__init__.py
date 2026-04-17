# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.safety_cert public API surface

"""sc_neurocore.safety_cert — IEC 61508 / ISO 26262 / FDA Class III safety certification.

Tier: industrial.

Two modules:

- ``safety_cert`` — automated certification artefact generators
  (traceability matrix, FMEDA, formal-proof certificates, WCET
  analysis, compliance checklists, certification package, CCF +
  proof-test + HFT assessments, IEC 62304 software safety
  classification, change-impact tracker, cross-standard mapper,
  formal-property gap detector, reliability metrics, evidence bag,
  safety-manual generator).
- ``safety_monitor`` — software-in-the-loop mirror of the
  formally-proven hardware safety monitor (6 properties).
"""

from sc_neurocore.safety_cert.safety_cert import (
    ASILLevel,
    CCFAnalysis,
    CCFDefence,
    CertificationGenerator,
    CertificationPackage,
    ChangeImpactTracker,
    ChangeRecord,
    ChecklistItem,
    ComplianceChecklist,
    CrossStandardMapper,
    EvidenceBag,
    EvidenceItem,
    FMEDA,
    FailureCategory,
    FailureMode,
    FormalProofCertificate,
    FormalProperty,
    FormalPropertyGapDetector,
    HFTAssessment,
    HFTLevel,
    IEC62304Assessment,
    ProofTestCoverage,
    PropertyGap,
    ReliabilityMetrics,
    Requirement,
    SafetyManualGenerator,
    SafetyStandard,
    SILLevel,
    SWClass,
    TraceabilityMatrix,
    WCETAnalyzer,
    WCETPath,
)
from sc_neurocore.safety_cert.safety_monitor import (
    SafetyLimits,
    SafetyMonitor,
)

__tier__ = "industrial"

__all__ = [
    # safety_cert.py — enums
    "ASILLevel",
    "FailureCategory",
    "HFTLevel",
    "SafetyStandard",
    "SILLevel",
    "SWClass",
    # safety_cert.py — dataclasses
    "CCFDefence",
    "CertificationPackage",
    "ChangeRecord",
    "ChecklistItem",
    "EvidenceItem",
    "FailureMode",
    "FormalProofCertificate",
    "FormalProperty",
    "HFTAssessment",
    "PropertyGap",
    "ReliabilityMetrics",
    "Requirement",
    "WCETPath",
    # safety_cert.py — generators / analysers
    "CCFAnalysis",
    "CertificationGenerator",
    "ChangeImpactTracker",
    "ComplianceChecklist",
    "CrossStandardMapper",
    "EvidenceBag",
    "FMEDA",
    "FormalPropertyGapDetector",
    "IEC62304Assessment",
    "ProofTestCoverage",
    "SafetyManualGenerator",
    "TraceabilityMatrix",
    "WCETAnalyzer",
    # safety_monitor.py
    "SafetyLimits",
    "SafetyMonitor",
]
