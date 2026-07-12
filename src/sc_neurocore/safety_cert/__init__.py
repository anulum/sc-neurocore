# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.safety_cert public API surface

"""Safety-evidence organisation and runtime monitoring.

The evidence modules build fail-closed traceability, FMEDA, formal-property,
timing, checklist, and manifest artifacts from explicit caller inputs. They do
not issue a certificate or replace a qualified conformity assessment.

The separate safety-monitor module is a software-in-the-loop mirror of the
hardware safety monitor.
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
