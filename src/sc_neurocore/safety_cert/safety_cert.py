# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Automated Safety & Regulatory Certification Generator

"""Auto-generates IEC 61508 / ISO 26262 / FDA Class III compliance artifacts.

Leverages SC-NeuroCore's deterministic bitstreams, SymbiYosys formal
proofs (7 modules / 72 properties), and traceable execution paths to
produce a one-click certification package:

- **Traceability Matrix**: Requirement → implementation → verification
- **FMEDA**: Failure Modes, Effects, and Diagnostic Analysis for SC modules
- **Formal Proof Certificates**: SymbiYosys results → safety-case evidence
- **WCET Analysis**: Worst-case execution time for SC bitstream paths
- **Safety Integrity Level (SIL) Assessment**: SIL 1–4 / ASIL A–D mapping
- **Compliance Checklist**: Standard-specific requirement coverage

Compatible with:
- ``hdl/formal/`` — SymbiYosys formal proofs (.sby configs)
- ``analysis/`` — spike statistics and correlation metrics
- ``uvm_gen/`` — UVM verification evidence
- ``sc_scope/`` — runtime monitoring evidence
"""

from __future__ import annotations

import hashlib
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ── Safety Standards ─────────────────────────────────────────────────


class SafetyStandard(Enum):
    IEC_61508 = "IEC 61508"
    ISO_26262 = "ISO 26262"
    FDA_CLASS_III = "FDA Class III"
    DO_254 = "DO-254"
    EN_50129 = "EN 50129"


class SILLevel(Enum):
    SIL_1 = 1
    SIL_2 = 2
    SIL_3 = 3
    SIL_4 = 4


class ASILLevel(Enum):
    QM = "QM"
    ASIL_A = "A"
    ASIL_B = "B"
    ASIL_C = "C"
    ASIL_D = "D"


SIL_TO_ASIL = {
    SILLevel.SIL_1: ASILLevel.ASIL_A,
    SILLevel.SIL_2: ASILLevel.ASIL_B,
    SILLevel.SIL_3: ASILLevel.ASIL_C,
    SILLevel.SIL_4: ASILLevel.ASIL_D,
}


# ── Requirement ──────────────────────────────────────────────────────


@dataclass
class Requirement:
    """One safety requirement with traceability links."""

    req_id: str
    description: str
    standard: SafetyStandard
    sil_level: SILLevel = SILLevel.SIL_2
    implementation_refs: List[str] = field(default_factory=list)
    verification_refs: List[str] = field(default_factory=list)
    status: str = "open"


# ── Traceability Matrix ─────────────────────────────────────────────


class TraceabilityMatrix:
    """Requirement → implementation → verification traceability.

    Each requirement links to:
    - Implementation artifacts (RTL files, Python modules)
    - Verification artifacts (formal proofs, UVM results, tests)
    """

    def __init__(self):
        self.requirements: Dict[str, Requirement] = {}

    def add_requirement(self, req: Requirement) -> None:
        self.requirements[req.req_id] = req

    def link_implementation(self, req_id: str, impl_ref: str) -> bool:
        req = self.requirements.get(req_id)
        if req is None:
            return False
        req.implementation_refs.append(impl_ref)
        self._update_status(req)
        return True

    def link_verification(self, req_id: str, verif_ref: str) -> bool:
        req = self.requirements.get(req_id)
        if req is None:
            return False
        req.verification_refs.append(verif_ref)
        self._update_status(req)
        return True

    def _update_status(self, req: Requirement) -> None:
        if req.implementation_refs and req.verification_refs:
            req.status = "verified"
        elif req.implementation_refs:
            req.status = "implemented"

    @property
    def coverage(self) -> float:
        if not self.requirements:
            return 0.0
        verified = sum(1 for r in self.requirements.values() if r.status == "verified")
        return verified / len(self.requirements)

    @property
    def open_count(self) -> int:
        return sum(1 for r in self.requirements.values() if r.status == "open")

    def generate_report(self) -> str:
        """Generate text traceability report."""
        lines = [
            "# Safety Traceability Matrix",
            f"Generated: {datetime.now().isoformat()}",
            f"Coverage: {self.coverage:.1%} ({len(self.requirements) - self.open_count}/{len(self.requirements)})",
            "",
            "| Req ID | Standard | SIL | Status | Impl | Verif |",
            "|--------|----------|-----|--------|------|-------|",
        ]
        for req in self.requirements.values():
            lines.append(
                f"| {req.req_id} | {req.standard.value} | SIL {req.sil_level.value} "
                f"| {req.status} | {len(req.implementation_refs)} | {len(req.verification_refs)} |"
            )
        return "\n".join(lines)


# ── FMEDA ────────────────────────────────────────────────────────────


class FailureCategory(Enum):
    SAFE = "safe"
    DANGEROUS_DETECTED = "dangerous_detected"
    DANGEROUS_UNDETECTED = "dangerous_undetected"
    NO_EFFECT = "no_effect"


@dataclass
class FailureMode:
    """One failure mode in the FMEDA."""

    fm_id: str
    component: str
    description: str
    category: FailureCategory
    failure_rate_fit: float  # FIT = failures per 10^9 hours
    diagnostic_coverage: float = 0.0  # 0.0 – 1.0
    mitigation: str = ""

    @property
    def safe_failure_fraction(self) -> float:
        """Fraction of failures that are safe or detected dangerous."""
        if self.category in (FailureCategory.SAFE, FailureCategory.NO_EFFECT):
            return 1.0
        if self.category == FailureCategory.DANGEROUS_DETECTED:
            return self.diagnostic_coverage
        return 0.0


class FMEDA:
    """Failure Modes, Effects, and Diagnostic Analysis.

    Aggregates failure modes for SC neuromorphic modules and computes
    Safe Failure Fraction (SFF) and Diagnostic Coverage (DC).
    """

    def __init__(self):
        self.failure_modes: List[FailureMode] = []

    def add_failure_mode(self, fm: FailureMode) -> None:
        self.failure_modes.append(fm)

    def add_sc_standard_modes(self, component: str) -> None:
        """Add standard SC-specific failure modes for a component."""
        modes = [
            FailureMode(
                f"{component}_LFSR_STUCK",
                component,
                "LFSR generator stuck at fixed value",
                FailureCategory.DANGEROUS_DETECTED,
                10.0,
                0.99,
                "ScDoctor ECC detects via Hamming check",
            ),
            FailureMode(
                f"{component}_BIT_FLIP",
                component,
                "Single-event upset in bitstream register",
                FailureCategory.DANGEROUS_DETECTED,
                50.0,
                0.95,
                "Bitstream correlation monitor detects SCC deviation",
            ),
            FailureMode(
                f"{component}_CLOCK_DRIFT",
                component,
                "Clock frequency deviation exceeds tolerance",
                FailureCategory.DANGEROUS_DETECTED,
                5.0,
                0.90,
                "Watchdog timer and SCC monitor",
            ),
            FailureMode(
                f"{component}_WEIGHT_CORRUPT",
                component,
                "Q8.8 weight corruption in BRAM",
                FailureCategory.DANGEROUS_DETECTED,
                20.0,
                0.98,
                "Formal proof guarantees range [0, w_max]",
            ),
            FailureMode(
                f"{component}_SAFE_SILENT",
                component,
                "Neuron fails to spike (silent failure)",
                FailureCategory.SAFE,
                30.0,
                1.0,
                "Firing rate monitor detects rate anomaly",
            ),
        ]
        self.failure_modes.extend(modes)

    @property
    def total_failure_rate(self) -> float:
        return sum(fm.failure_rate_fit for fm in self.failure_modes)

    @property
    def safe_failure_fraction(self) -> float:
        """SFF = (safe + no_effect + DC*dangerous_detected) / total."""
        if not self.failure_modes:
            return 0.0
        total = self.total_failure_rate
        if total == 0:
            return 0.0
        safe_sum = sum(fm.failure_rate_fit * fm.safe_failure_fraction for fm in self.failure_modes)
        return safe_sum / total

    @property
    def diagnostic_coverage(self) -> float:
        dd = [fm for fm in self.failure_modes if fm.category == FailureCategory.DANGEROUS_DETECTED]
        if not dd:
            return 0.0
        return sum(fm.diagnostic_coverage * fm.failure_rate_fit for fm in dd) / sum(
            fm.failure_rate_fit for fm in dd
        )

    @property
    def residual_risk_fit(self) -> float:
        """Dangerous-undetected failure rate (residual risk)."""
        return sum(
            fm.failure_rate_fit * (1.0 - fm.safe_failure_fraction) for fm in self.failure_modes
        )

    def sff_by_component(self) -> Dict[str, float]:
        """Per-component safe failure fraction."""
        components: Dict[str, List[FailureMode]] = {}
        for fm in self.failure_modes:
            components.setdefault(fm.component, []).append(fm)
        result = {}
        for comp, fms in components.items():
            total = sum(f.failure_rate_fit for f in fms)
            safe = sum(f.failure_rate_fit * f.safe_failure_fraction for f in fms)
            result[comp] = safe / total if total > 0 else 0.0
        return result

    def max_achievable_sil(self) -> SILLevel:
        """Determine max achievable SIL from SFF and DC."""
        sff = self.safe_failure_fraction
        dc = self.diagnostic_coverage
        if sff >= 0.99 and dc >= 0.99:
            return SILLevel.SIL_4
        if sff >= 0.97 and dc >= 0.99:
            return SILLevel.SIL_3
        if sff >= 0.90 and dc >= 0.90:
            return SILLevel.SIL_2
        if sff >= 0.60:
            return SILLevel.SIL_1
        return SILLevel.SIL_1

    def generate_report(self) -> str:
        lines = [
            "# FMEDA Report",
            f"Total failure rate: {self.total_failure_rate:.1f} FIT",
            f"Safe Failure Fraction: {self.safe_failure_fraction:.1%}",
            f"Diagnostic Coverage: {self.diagnostic_coverage:.1%}",
            f"Max achievable SIL: SIL {self.max_achievable_sil().value}",
            "",
            "| FM ID | Component | Category | Rate (FIT) | DC | Mitigation |",
            "|-------|-----------|----------|------------|-----|------------|",
        ]
        for fm in self.failure_modes:
            lines.append(
                f"| {fm.fm_id} | {fm.component} | {fm.category.value} "
                f"| {fm.failure_rate_fit:.1f} | {fm.diagnostic_coverage:.0%} | {fm.mitigation} |"
            )
        return "\n".join(lines)


# ── Formal Proof Certificate ────────────────────────────────────────


@dataclass
class FormalProperty:
    """One formally verified property."""

    prop_id: str
    module: str
    description: str
    property_type: str  # "assert" | "cover" | "assume"
    status: str = "unknown"  # "proven" | "failed" | "unknown"
    engine: str = "SymbiYosys"
    depth: int = 20
    sby_file: str = ""


@dataclass
class FormalProofCertificate:
    """Certificate packaging formal verification evidence."""

    properties: List[FormalProperty] = field(default_factory=list)
    generation_timestamp: str = ""
    tool_version: str = "SymbiYosys"
    certificate_hash: str = ""

    def add_property(self, prop: FormalProperty) -> None:
        self.properties.append(prop)

    @property
    def proven_count(self) -> int:
        return sum(1 for p in self.properties if p.status == "proven")

    @property
    def total_count(self) -> int:
        return len(self.properties)

    @property
    def pass_rate(self) -> float:
        return self.proven_count / self.total_count if self.total_count > 0 else 0.0

    def compute_hash(self) -> str:
        h = hashlib.sha256()
        for p in sorted(self.properties, key=lambda x: x.prop_id):
            h.update(f"{p.prop_id}:{p.status}:{p.module}".encode())
        self.certificate_hash = h.hexdigest()[:32]
        self.generation_timestamp = datetime.now().isoformat()
        return self.certificate_hash

    def generate_report(self) -> str:
        lines = [
            "# Formal Proof Certificate",
            f"Generated: {self.generation_timestamp or datetime.now().isoformat()}",
            f"Hash: {self.certificate_hash or self.compute_hash()}",
            f"Tool: {self.tool_version}",
            f"Properties: {self.proven_count}/{self.total_count} proven ({self.pass_rate:.0%})",
            "",
            "| Property | Module | Type | Status | Depth |",
            "|----------|--------|------|--------|-------|",
        ]
        for p in self.properties:
            lines.append(
                f"| {p.prop_id} | {p.module} | {p.property_type} | {p.status} | {p.depth} |"
            )
        return "\n".join(lines)


# ── WCET Analysis ───────────────────────────────────────────────────


@dataclass
class WCETPath:
    """Worst-case execution time for one SC computation path."""

    path_id: str
    description: str
    stages: List[str]
    cycles_per_stage: List[int]

    @property
    def total_cycles(self) -> int:
        return sum(self.cycles_per_stage)

    def wcet_ns(self, clock_mhz: float) -> float:
        return self.total_cycles * 1000.0 / clock_mhz


class WCETAnalyzer:
    """Worst-case execution time analysis for SC bitstream paths.

    Uses static analysis of the SC pipeline stages:
    - LFSR encoding: bitstream_length cycles
    - Dot product: num_inputs cycles
    - LIF evaluation: fixed (3 cycles)
    - AER encoding: num_neurons worst-case
    - STP update: 1 cycle
    """

    LFSR_OVERHEAD = 1
    DOT_PRODUCT_PER_INPUT = 1
    LIF_FIXED = 3
    AER_PER_NEURON = 1
    STP_FIXED = 1

    @classmethod
    def analyze(
        cls,
        bitstream_length: int,
        num_inputs: int,
        num_neurons: int,
        has_stp: bool = False,
    ) -> WCETPath:
        stages = ["LFSR_Encode", "DotProduct", "LIF_Eval", "AER_Encode"]
        cycles = [
            bitstream_length * cls.LFSR_OVERHEAD,
            num_inputs * cls.DOT_PRODUCT_PER_INPUT,
            cls.LIF_FIXED,
            num_neurons * cls.AER_PER_NEURON,
        ]
        if has_stp:
            stages.append("STP_Update")
            cycles.append(cls.STP_FIXED)
        return WCETPath(
            path_id="sc_inference",
            description="Full SC inference pipeline",
            stages=stages,
            cycles_per_stage=cycles,
        )

    @classmethod
    def analyze_multistage(
        cls,
        layers: List[Dict[str, int]],
    ) -> WCETPath:
        """Analyze a multi-layer SC network."""
        stages = []
        cycles = []
        for i, layer in enumerate(layers):
            bs = layer.get("bitstream_length", 256)
            ni = layer.get("num_inputs", 8)
            nn = layer.get("num_neurons", 16)
            stages.extend([f"L{i}_LFSR", f"L{i}_Dot", f"L{i}_LIF", f"L{i}_AER"])
            cycles.extend(
                [
                    bs * cls.LFSR_OVERHEAD,
                    ni * cls.DOT_PRODUCT_PER_INPUT,
                    cls.LIF_FIXED,
                    nn * cls.AER_PER_NEURON,
                ]
            )
        return WCETPath("sc_network", "Multi-layer SC network", stages, cycles)


# ── Compliance Checklist ─────────────────────────────────────────────


@dataclass
class ChecklistItem:
    """One item in a compliance checklist."""

    item_id: str
    clause: str
    description: str
    evidence: str = ""
    status: str = "not_addressed"  # "compliant" | "partial" | "not_addressed"


class ComplianceChecklist:
    """Standard-specific compliance checklist generator."""

    IEC_61508_CLAUSES = [
        ("7.4.2", "Formal verification of safety functions", "formal/"),
        ("7.4.3", "Semi-formal methods for SIL3+", "uvm_gen/"),
        ("7.4.4", "Modular design and coding standards", "hdl/"),
        ("7.4.5", "Structured testing", "tb/"),
        ("7.4.7", "Impact analysis of modifications", "analysis/"),
        ("7.6.2", "Hardware diagnostic coverage", "sc_scope/"),
        ("7.9.2", "Safety validation tests", "formal/ + uvm_gen/"),
    ]

    ISO_26262_CLAUSES = [
        ("6.4.1", "Hardware-software interface specification", "hdl/sc_axil_cfg.v"),
        ("6.4.2", "Safety analysis (FMEDA)", "safety_cert/fmeda"),
        ("6.4.3", "Dependent failures analysis", "formal/"),
        ("6.7.4", "Verification of safety requirements", "formal/ + uvm_gen/"),
        ("8.4.3", "Unit testing of software", "tests/"),
        ("8.4.5", "Back-to-back comparison testing", "sc_scope/"),
        ("9.4.1", "Functional safety concept verification", "analysis/"),
    ]

    FDA_CLASS_III_CLAUSES = [
        ("820.30", "Design controls and verification", "formal/ + uvm_gen/"),
        ("820.30(g)", "Design validation", "tb/"),
        ("820.70", "Production and process controls", "asic_flow/"),
        ("820.72", "Inspection, measuring, and test equipment", "sc_scope/"),
        ("820.90", "Nonconforming product handling", "fault_injection/"),
        ("Pre-Submission", "Software documentation (IEC 62304)", "docs/"),
        ("510(k)", "Substantial equivalence demonstration", "analysis/"),
    ]

    DO_254_CLAUSES = [
        ("4.0", "Planning process (PHAC)", "docs/"),
        ("5.0", "Hardware design process", "hdl/"),
        ("6.0", "Validation and verification", "formal/ + uvm_gen/"),
        ("7.0", "Configuration management", ".git/ + CI_ECOSYSTEM"),
        ("8.0", "Process assurance", "analysis/"),
        ("9.0", "Certification liaison", "safety_cert/"),
    ]

    EN_50129_CLAUSES = [
        ("5.3", "Safety integrity requirements", "safety_cert/fmeda"),
        ("5.4", "Technical safety report", "safety_cert/"),
        ("6.1", "Quality management", "CI_ECOSYSTEM"),
        ("6.2", "Safety management", "safety_cert/"),
        ("7.2", "Evidence of safety validation", "formal/ + tb/"),
        ("7.3", "Evidence of functional safety", "sc_scope/ + fault_injection/"),
    ]

    @classmethod
    def generate(cls, standard: SafetyStandard) -> List[ChecklistItem]:
        clause_map = {
            SafetyStandard.IEC_61508: cls.IEC_61508_CLAUSES,
            SafetyStandard.ISO_26262: cls.ISO_26262_CLAUSES,
            SafetyStandard.FDA_CLASS_III: cls.FDA_CLASS_III_CLAUSES,
            SafetyStandard.DO_254: cls.DO_254_CLAUSES,
            SafetyStandard.EN_50129: cls.EN_50129_CLAUSES,
        }
        clauses = clause_map.get(standard, [])
        items = []
        for clause, desc, evidence in clauses:
            items.append(
                ChecklistItem(
                    item_id=f"{standard.value}_{clause}",
                    clause=clause,
                    description=desc,
                    evidence=evidence,
                    status="partial" if evidence else "not_addressed",
                )
            )
        return items


# ── Certification Package Generator ──────────────────────────────────


@dataclass
class CertificationPackage:
    """Complete certification package output."""

    standard: SafetyStandard
    sil_level: SILLevel
    traceability_report: str
    fmeda_report: str
    formal_cert_report: str
    wcet_report: str
    checklist: List[ChecklistItem]
    package_hash: str = ""
    generated: str = ""

    @property
    def checklist_coverage(self) -> float:
        if not self.checklist:
            return 0.0
        addressed = sum(1 for c in self.checklist if c.status != "not_addressed")
        return addressed / len(self.checklist)


class CertificationGenerator:
    """Top-level generator for safety certification packages."""

    def generate(
        self,
        standard: SafetyStandard,
        target_sil: SILLevel,
        modules: List[str],
        formal_properties: List[FormalProperty],
        network_config: Optional[Dict[str, int]] = None,
    ) -> CertificationPackage:
        """Generate a complete certification package."""
        # 1. Traceability
        tm = TraceabilityMatrix()
        for i, mod in enumerate(modules):
            req = Requirement(
                req_id=f"REQ_{i + 1:03d}",
                description=f"Safety function for {mod}",
                standard=standard,
                sil_level=target_sil,
                implementation_refs=[f"hdl/{mod}.v"],
            )
            tm.add_requirement(req)
            matching = [p for p in formal_properties if p.module == mod and p.status == "proven"]
            for p in matching:
                tm.link_verification(req.req_id, p.sby_file or p.prop_id)

        # 2. FMEDA
        fmeda = FMEDA()
        for mod in modules:
            fmeda.add_sc_standard_modes(mod)

        # 3. Formal certificate
        cert = FormalProofCertificate(properties=list(formal_properties))
        cert.compute_hash()

        # 4. WCET
        cfg = network_config or {"bitstream_length": 256, "num_inputs": 8, "num_neurons": 16}
        wcet = WCETAnalyzer.analyze(
            cfg.get("bitstream_length", 256),
            cfg.get("num_inputs", 8),
            cfg.get("num_neurons", 16),
        )
        clock = cfg.get("clock_mhz", 100)
        wcet_text = (
            f"WCET: {wcet.total_cycles} cycles = {wcet.wcet_ns(clock):.1f} ns "
            f"@ {clock} MHz\nStages: {' → '.join(wcet.stages)}"
        )

        # 5. Checklist
        checklist = ComplianceChecklist.generate(standard)

        # 6. Package hash
        h = hashlib.sha256()
        h.update(cert.certificate_hash.encode())
        h.update(standard.value.encode())
        h.update(str(target_sil.value).encode())
        pkg_hash = h.hexdigest()[:32]

        return CertificationPackage(
            standard=standard,
            sil_level=target_sil,
            traceability_report=tm.generate_report(),
            fmeda_report=fmeda.generate_report(),
            formal_cert_report=cert.generate_report(),
            wcet_report=wcet_text,
            checklist=checklist,
            package_hash=pkg_hash,
            generated=datetime.now().isoformat(),
        )


# ── Common Cause Failure (CCF) Analysis (Gap 1) ─────────────────────


@dataclass
class CCFDefence:
    """One defence against common cause failures."""

    defence_id: str
    description: str
    category: str  # "separation", "diversity", "complexity", "assessment", "competence"
    beta_reduction: float = 0.0  # Reduction in β-factor
    implemented: bool = False


class CCFAnalysis:
    """IEC 61508 Annex D — β-factor estimation for common cause failures.

    β = fraction of dangerous failures that are common cause.
    Defences reduce β. Lower β → higher achievable SIL.
    """

    DEFAULT_DEFENCES = [
        CCFDefence("D1", "Physical separation of redundant channels", "separation", 0.025),
        CCFDefence("D2", "Diverse hardware implementations", "diversity", 0.02),
        CCFDefence("D3", "Diverse software / bitstream encodings", "diversity", 0.02),
        CCFDefence("D4", "Independent design teams", "competence", 0.01),
        CCFDefence("D5", "Environmental conditioning/shielding", "separation", 0.015),
        CCFDefence("D6", "Complexity analysis and minimisation", "complexity", 0.01),
    ]

    def __init__(self):
        self.defences: List[CCFDefence] = [
            CCFDefence(d.defence_id, d.description, d.category, d.beta_reduction, d.implemented)
            for d in self.DEFAULT_DEFENCES
        ]

    def mark_implemented(self, defence_id: str) -> bool:
        for d in self.defences:
            if d.defence_id == defence_id:
                d.implemented = True
                return True
        return False

    @property
    def beta_factor(self) -> float:
        """Resulting β-factor (start at 0.10, reduce by implemented defences)."""
        base = 0.10
        reduction = sum(d.beta_reduction for d in self.defences if d.implemented)
        return max(0.005, base - reduction)

    @property
    def implemented_count(self) -> int:
        return sum(1 for d in self.defences if d.implemented)

    def sil_compatible(self, target_sil: SILLevel) -> bool:
        """Check if β is low enough for the target SIL."""
        thresholds = {
            SILLevel.SIL_1: 0.10,
            SILLevel.SIL_2: 0.05,
            SILLevel.SIL_3: 0.02,
            SILLevel.SIL_4: 0.01,
        }
        return self.beta_factor <= thresholds.get(target_sil, 0.10)


# ── Proof-of-Test Coverage (Gap 2) ──────────────────────────────────


class ProofTestCoverage:
    """Maps formal proof pass/fail to diagnostic coverage for SIL claims."""

    @staticmethod
    def coverage_from_proofs(properties: List[FormalProperty]) -> float:
        """Formal proof coverage = proven / total asserts."""
        asserts = [p for p in properties if p.property_type == "assert"]
        if not asserts:
            return 0.0
        proven = sum(1 for p in asserts if p.status == "proven")
        return proven / len(asserts)

    @staticmethod
    def dc_to_sil(dc: float) -> SILLevel:
        """Map diagnostic coverage to max SIL per IEC 61508 Table 3."""
        if dc >= 0.99:
            return SILLevel.SIL_4
        if dc >= 0.99:
            return SILLevel.SIL_3
        if dc >= 0.90:
            return SILLevel.SIL_2
        if dc >= 0.60:
            return SILLevel.SIL_1
        return SILLevel.SIL_1

    @staticmethod
    def uncovered_modules(properties: List[FormalProperty], all_modules: List[str]) -> List[str]:
        """Modules with no formal proofs."""
        covered = {p.module for p in properties}
        return [m for m in all_modules if m not in covered]


# ── Hardware Fault Tolerance (Gap 3) ────────────────────────────────


class HFTLevel(Enum):
    HFT_0 = 0
    HFT_1 = 1
    HFT_2 = 2


@dataclass
class HFTAssessment:
    """Hardware Fault Tolerance assessment per IEC 61508 Table 2."""

    sff: float
    target_sil: SILLevel

    @property
    def required_hft(self) -> HFTLevel:
        """Determine required HFT from SFF and target SIL."""
        if self.sff >= 0.99:
            if self.target_sil.value <= 3:
                return HFTLevel.HFT_0
            return HFTLevel.HFT_1
        elif self.sff >= 0.90:
            if self.target_sil.value <= 2:
                return HFTLevel.HFT_0
            elif self.target_sil.value == 3:
                return HFTLevel.HFT_1
            return HFTLevel.HFT_2
        elif self.sff >= 0.60:
            if self.target_sil.value <= 1:
                return HFTLevel.HFT_0
            elif self.target_sil.value == 2:
                return HFTLevel.HFT_1
            return HFTLevel.HFT_2
        else:
            if self.target_sil.value <= 1:
                return HFTLevel.HFT_1
            return HFTLevel.HFT_2

    @property
    def is_simplex_ok(self) -> bool:
        return self.required_hft == HFTLevel.HFT_0


# ── Change Impact Analysis (Gap 4) ──────────────────────────────────


@dataclass
class ChangeRecord:
    """One tracked change affecting safety."""

    change_id: str
    description: str
    affected_modules: List[str]
    affected_reqs: List[str]
    risk_level: str = "low"  # "low", "medium", "high"
    re_verification_needed: bool = False


class ChangeImpactTracker:
    """Tracks changes and their impact on certification artifacts."""

    def __init__(self):
        self.changes: List[ChangeRecord] = []

    def add_change(self, change: ChangeRecord) -> None:
        if change.risk_level in ("medium", "high"):
            change.re_verification_needed = True
        self.changes.append(change)

    def affected_requirements(self) -> List[str]:
        reqs: set = set()
        for c in self.changes:
            reqs.update(c.affected_reqs)
        return sorted(reqs)

    @property
    def high_risk_count(self) -> int:
        return sum(1 for c in self.changes if c.risk_level == "high")

    @property
    def needs_re_certification(self) -> bool:
        return self.high_risk_count > 0


# ── Safety Manual Template (Gap 5) ──────────────────────────────────


class SafetyManualGenerator:
    """Generates operator safety manual skeleton per IEC 61508-2 §7.4.15."""

    @staticmethod
    def generate(
        product_name: str,
        sil_level: SILLevel,
        modules: List[str],
        wcet_ns: float,
    ) -> str:
        return textwrap.dedent(f"""\
# Safety Manual — {product_name}
## SIL {sil_level.value} — {datetime.now().strftime("%Y-%m-%d")}

### 1. Product Description
SC-NeuroCore neuromorphic processor implementing {len(modules)} safety-related modules.

### 2. Safety Functions
{chr(10).join(f"- {m}" for m in modules)}

### 3. Operating Conditions
- Supply voltage: per PDK specification
- Clock: configured at synthesis
- WCET (worst-case): {wcet_ns:.1f} ns

### 4. Safety Integrity Level
- Target: SIL {sil_level.value} (continuous mode)
- ASIL equivalent: {SIL_TO_ASIL.get(sil_level, ASILLevel.QM).value}

### 5. Proof Test Interval
- Recommended: per FMEDA residual risk analysis
- Online diagnostics: bitstream correlation monitor, ECC, watchdog

### 6. Limitations
- SC bitstream length must match design specification
- Environmental temperature must remain within PDK limits
- Modifications require change impact re-assessment

### 7. Maintenance
- Firmware updates require re-verification of formal proofs
- Annual proof test of all safety functions recommended
""")


# ── IEC 62304 Software Lifecycle Class (Gap 6) ──────────────────────


class SWClass(Enum):
    CLASS_A = "A"  # No injury possible
    CLASS_B = "B"  # Non-serious injury possible
    CLASS_C = "C"  # Death or serious injury possible


@dataclass
class IEC62304Assessment:
    """IEC 62304 software safety classification for medical devices."""

    sw_class: SWClass
    hazard_description: str = ""
    risk_control_measures: List[str] = field(default_factory=list)

    @property
    def requires_unit_testing(self) -> bool:
        return self.sw_class in (SWClass.CLASS_B, SWClass.CLASS_C)

    @property
    def requires_architectural_design(self) -> bool:
        return self.sw_class == SWClass.CLASS_C

    @staticmethod
    def from_sil(sil: SILLevel) -> IEC62304Assessment:
        mapping = {
            SILLevel.SIL_1: SWClass.CLASS_A,
            SILLevel.SIL_2: SWClass.CLASS_B,
            SILLevel.SIL_3: SWClass.CLASS_C,
            SILLevel.SIL_4: SWClass.CLASS_C,
        }
        return IEC62304Assessment(sw_class=mapping.get(sil, SWClass.CLASS_A))


# ── MTBF / Reliability Calculation (Gap 7) ──────────────────────────


@dataclass
class ReliabilityMetrics:
    """System-level reliability from FMEDA data."""

    total_fit: float
    dangerous_undetected_fit: float

    @property
    def mtbf_hours(self) -> float:
        """Mean Time Between Failures (hours)."""
        if self.total_fit <= 0:
            return float("inf")
        return 1e9 / self.total_fit

    @property
    def mtbf_years(self) -> float:
        return self.mtbf_hours / 8760.0

    @property
    def pfh_d(self) -> float:
        """Probability of dangerous failure per hour."""
        if self.dangerous_undetected_fit <= 0:
            return 0.0
        return self.dangerous_undetected_fit / 1e9

    @property
    def pfh_sil(self) -> SILLevel:
        """Max SIL from PFHd per IEC 61508 Table 3."""
        pfh = self.pfh_d
        if pfh <= 1e-8:
            return SILLevel.SIL_4
        if pfh <= 1e-7:
            return SILLevel.SIL_3
        if pfh <= 1e-6:
            return SILLevel.SIL_2
        return SILLevel.SIL_1

    @staticmethod
    def from_fmeda(fmeda: FMEDA) -> ReliabilityMetrics:
        return ReliabilityMetrics(
            total_fit=fmeda.total_failure_rate,
            dangerous_undetected_fit=fmeda.residual_risk_fit,
        )


# ── Evidence Bag Manifest (Gap 8) ───────────────────────────────────


@dataclass
class EvidenceItem:
    """One item in the evidence bag."""

    filename: str
    category: str  # "formal", "test", "analysis", "design", "report"
    description: str
    sha256: str = ""


class EvidenceBag:
    """Manifest of all certification evidence artifacts."""

    def __init__(self):
        self.items: List[EvidenceItem] = []

    def add(self, item: EvidenceItem) -> None:
        self.items.append(item)

    def add_from_package(self, pkg: CertificationPackage) -> None:
        self.add(EvidenceItem("traceability_matrix.md", "report", "Requirement traceability"))
        self.add(EvidenceItem("fmeda_report.md", "analysis", "FMEDA analysis"))
        self.add(EvidenceItem("formal_proof_cert.md", "formal", "Formal proof certificate"))
        self.add(EvidenceItem("wcet_analysis.md", "analysis", "WCET analysis"))
        self.add(EvidenceItem("compliance_checklist.md", "report", "Compliance checklist"))

    @property
    def file_count(self) -> int:
        return len(self.items)

    def manifest(self) -> str:
        lines = ["# Evidence Bag Manifest", f"Items: {self.file_count}", ""]
        for item in self.items:
            lines.append(f"- [{item.category}] {item.filename}: {item.description}")
        return "\n".join(lines)

    def compute_hashes(self) -> str:
        h = hashlib.sha256()
        for item in sorted(self.items, key=lambda x: x.filename):
            h.update(f"{item.filename}:{item.category}".encode())
        return h.hexdigest()[:32]


# ── Multi-Standard Cross-Mapping (Gap 9) ────────────────────────────

CROSS_MAP = {
    ("IEC 61508", "7.4.2"): [("ISO 26262", "6.7.4"), ("DO-254", "6.0")],
    ("IEC 61508", "7.4.3"): [("ISO 26262", "6.4.1")],
    ("IEC 61508", "7.6.2"): [("ISO 26262", "6.4.2"), ("FDA Class III", "820.72")],
    ("IEC 61508", "7.9.2"): [("ISO 26262", "9.4.1"), ("FDA Class III", "820.30")],
    ("ISO 26262", "8.4.3"): [("FDA Class III", "820.30(g)")],
}


class CrossStandardMapper:
    """Maps equivalent clauses across IEC 61508 / ISO 26262 / FDA / DO-254."""

    @staticmethod
    def equivalent_clauses(standard: str, clause: str) -> List[Tuple[str, str]]:
        return CROSS_MAP.get((standard, clause), [])

    @staticmethod
    def coverage_overlap(checklist_a: List[ChecklistItem], checklist_b: List[ChecklistItem]) -> int:
        """Count shared compliance coverage between two checklists."""
        addressed_a = {i.clause for i in checklist_a if i.status != "not_addressed"}
        addressed_b = {i.clause for i in checklist_b if i.status != "not_addressed"}
        shared = 0
        for std_a, clause_a in [
            (i.item_id.rsplit("_", 1)[0], i.clause) for i in checklist_a if i.clause in addressed_a
        ]:
            for mapping in CROSS_MAP.get((std_a, clause_a), []):
                if mapping[1] in addressed_b:
                    shared += 1
        return shared


# ── Formal Property Gap Detector (Gap 10) ───────────────────────────


@dataclass
class PropertyGap:
    """One module with insufficient formal coverage."""

    module: str
    total_properties: int
    proven_properties: int
    missing_types: List[str]  # property types not covered

    @property
    def coverage(self) -> float:
        return self.proven_properties / self.total_properties if self.total_properties > 0 else 0.0


class FormalPropertyGapDetector:
    """Detects modules with insufficient formal verification coverage."""

    REQUIRED_TYPES = ["assert", "cover"]

    @classmethod
    def detect(
        cls, properties: List[FormalProperty], required_modules: List[str]
    ) -> List[PropertyGap]:
        by_module: Dict[str, List[FormalProperty]] = {}
        for p in properties:
            by_module.setdefault(p.module, []).append(p)

        gaps = []
        for mod in required_modules:
            props = by_module.get(mod, [])
            proven = [p for p in props if p.status == "proven"]
            types_present = {p.property_type for p in props}
            missing = [t for t in cls.REQUIRED_TYPES if t not in types_present]

            if not props or len(proven) < len(props) or missing:
                gaps.append(
                    PropertyGap(
                        module=mod,
                        total_properties=len(props),
                        proven_properties=len(proven),
                        missing_types=missing,
                    )
                )
        return gaps

    @classmethod
    def is_fully_covered(
        cls, properties: List[FormalProperty], required_modules: List[str]
    ) -> bool:
        return len(cls.detect(properties, required_modules)) == 0
