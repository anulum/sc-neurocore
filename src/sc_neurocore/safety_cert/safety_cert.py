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
import math
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

    def __post_init__(self) -> None:
        if not isinstance(self.req_id, str) or not self.req_id.strip():
            raise ValueError("req_id must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if not isinstance(self.standard, SafetyStandard):
            raise ValueError("standard must be a SafetyStandard")
        if not isinstance(self.sil_level, SILLevel):
            raise ValueError("sil_level must be a SILLevel")
        if not isinstance(self.status, str) or self.status not in {
            "open",
            "implemented",
            "verified",
        }:
            raise ValueError("status must be one of: open, implemented, verified")

        for impl_ref in self.implementation_refs:
            if not isinstance(impl_ref, str) or not impl_ref.strip():
                raise ValueError("implementation_refs must contain non-empty strings")
        for verif_ref in self.verification_refs:
            if not isinstance(verif_ref, str) or not verif_ref.strip():
                raise ValueError("verification_refs must contain non-empty strings")


# ── Traceability Matrix ─────────────────────────────────────────────


class TraceabilityMatrix:
    """Requirement → implementation → verification traceability.

    Each requirement links to:
    - Implementation artifacts (RTL files, Python modules)
    - Verification artifacts (formal proofs, UVM results, tests)
    """

    def __init__(self) -> None:
        self.requirements: Dict[str, Requirement] = {}

    def add_requirement(self, req: Requirement) -> None:
        if not isinstance(req, Requirement):
            raise ValueError("req must be a Requirement")
        if req.req_id in self.requirements:
            raise ValueError(f"requirement already exists: {req.req_id}")
        self.requirements[req.req_id] = req

    def link_implementation(self, req_id: str, impl_ref: str) -> bool:
        if not isinstance(req_id, str) or not req_id.strip():
            raise ValueError("req_id must be a non-empty string")
        if not isinstance(impl_ref, str) or not impl_ref.strip():
            raise ValueError("impl_ref must be a non-empty string")
        req_id = req_id.strip()
        impl_ref = impl_ref.strip()
        req = self.requirements.get(req_id)
        if req is None:
            return False
        if not isinstance(req, Requirement):
            raise ValueError("requirements must contain Requirement entries")
        if impl_ref in req.implementation_refs:
            self._update_status(req)
            return True
        req.implementation_refs.append(impl_ref)
        self._update_status(req)
        return True

    def link_verification(self, req_id: str, verif_ref: str) -> bool:
        if not isinstance(req_id, str) or not req_id.strip():
            raise ValueError("req_id must be a non-empty string")
        if not isinstance(verif_ref, str) or not verif_ref.strip():
            raise ValueError("verif_ref must be a non-empty string")
        req_id = req_id.strip()
        verif_ref = verif_ref.strip()
        req = self.requirements.get(req_id)
        if req is None:
            return False
        if not isinstance(req, Requirement):
            raise ValueError("requirements must contain Requirement entries")
        if verif_ref in req.verification_refs:
            self._update_status(req)
            return True
        req.verification_refs.append(verif_ref)
        self._update_status(req)
        return True

    def _update_status(self, req: Requirement) -> None:
        if not isinstance(req, Requirement):
            raise ValueError("req must be a Requirement")
        for impl_ref in req.implementation_refs:
            if not isinstance(impl_ref, str) or not impl_ref.strip():
                raise ValueError("implementation_refs must contain non-empty strings")
        for verif_ref in req.verification_refs:
            if not isinstance(verif_ref, str) or not verif_ref.strip():
                raise ValueError("verification_refs must contain non-empty strings")
        if req.implementation_refs and req.verification_refs:
            req.status = "verified"
        elif req.implementation_refs:
            req.status = "implemented"
        else:
            req.status = "open"

    @property
    def coverage(self) -> float:
        if not self.requirements:
            return 0.0
        for req_key, req in self.requirements.items():
            if not isinstance(req, Requirement):
                raise ValueError("requirements must contain Requirement entries")
            if req.req_id != req_key:
                raise ValueError("requirement key mismatch with req_id")
            if not isinstance(req.status, str) or req.status not in {
                "open",
                "implemented",
                "verified",
            }:
                raise ValueError(
                    "requirements statuses must be one of: open, implemented, verified"
                )
            if not isinstance(req.standard, SafetyStandard):
                raise ValueError("requirements must use SafetyStandard")
            if not isinstance(req.sil_level, SILLevel):
                raise ValueError("requirements must use SILLevel")
        verified = sum(1 for r in self.requirements.values() if r.status == "verified")
        return verified / len(self.requirements)

    @property
    def open_count(self) -> int:
        for req_key, req in self.requirements.items():
            if not isinstance(req, Requirement):
                raise ValueError("requirements must contain Requirement entries")
            if req.req_id != req_key:
                raise ValueError("requirement key mismatch with req_id")
            if not isinstance(req.status, str) or req.status not in {
                "open",
                "implemented",
                "verified",
            }:
                raise ValueError(
                    "requirements statuses must be one of: open, implemented, verified"
                )
        return sum(1 for r in self.requirements.values() if r.status == "open")

    def generate_report(self) -> str:
        """Generate text traceability report."""
        for req_key, req in self.requirements.items():
            if not isinstance(req, Requirement):
                raise ValueError("requirements must contain Requirement entries")
            if req.req_id != req_key:
                raise ValueError("requirement key mismatch with req_id")
            if not isinstance(req.status, str) or req.status not in {
                "open",
                "implemented",
                "verified",
            }:
                raise ValueError(
                    "requirements statuses must be one of: open, implemented, verified"
                )
        lines = [
            "# Safety Traceability Matrix",
            f"Generated: {datetime.now().isoformat()}",
            f"Coverage: {self.coverage:.1%} ({len(self.requirements) - self.open_count}/{len(self.requirements)})",
            "",
            "| Req ID | Standard | SIL | Status | Impl | Verif |",
            "|--------|----------|-----|--------|------|-------|",
        ]
        for req in sorted(self.requirements.values(), key=lambda r: r.req_id):
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

    def __post_init__(self) -> None:
        if not isinstance(self.fm_id, str) or not self.fm_id.strip():
            raise ValueError("fm_id must be a non-empty string")
        if not isinstance(self.component, str) or not self.component.strip():
            raise ValueError("component must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if not isinstance(self.category, FailureCategory):
            raise ValueError("category must be a FailureCategory")
        if isinstance(self.failure_rate_fit, bool) or not isinstance(
            self.failure_rate_fit, int | float
        ):
            raise ValueError("failure_rate_fit must be a finite non-negative number")
        if not math.isfinite(float(self.failure_rate_fit)) or float(self.failure_rate_fit) < 0.0:
            raise ValueError("failure_rate_fit must be a finite non-negative number")
        if isinstance(self.diagnostic_coverage, bool) or not isinstance(
            self.diagnostic_coverage, int | float
        ):
            raise ValueError("diagnostic_coverage must be a finite value in [0, 1]")
        if (
            not math.isfinite(float(self.diagnostic_coverage))
            or float(self.diagnostic_coverage) < 0.0
            or float(self.diagnostic_coverage) > 1.0
        ):
            raise ValueError("diagnostic_coverage must be a finite value in [0, 1]")
        if not isinstance(self.mitigation, str):
            raise ValueError("mitigation must be a string")

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

    def __init__(self) -> None:
        self.failure_modes: List[FailureMode] = []

    def add_failure_mode(self, fm: FailureMode) -> None:
        if not isinstance(fm, FailureMode):
            raise ValueError("fm must be a FailureMode")
        if any(existing.fm_id == fm.fm_id for existing in self.failure_modes):
            raise ValueError(f"failure mode already exists: {fm.fm_id}")
        self.failure_modes.append(fm)

    def add_sc_standard_modes(self, component: str) -> None:
        """Add standard SC-specific failure modes for a component."""
        if not isinstance(component, str) or not component.strip():
            raise ValueError("component must be a non-empty string")
        component = component.strip()
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
        for mode in modes:
            self.add_failure_mode(mode)

    @property
    def total_failure_rate(self) -> float:
        for fm in self.failure_modes:
            if not isinstance(fm, FailureMode):
                raise ValueError("failure_modes must contain FailureMode entries")
            if not math.isfinite(float(fm.failure_rate_fit)) or float(fm.failure_rate_fit) < 0.0:
                raise ValueError("failure_rate_fit must be a finite non-negative value")
        total = sum(fm.failure_rate_fit for fm in self.failure_modes)
        if not math.isfinite(float(total)) or total < 0.0:
            raise ValueError("total_failure_rate must be a finite non-negative value")
        return total

    @property
    def safe_failure_fraction(self) -> float:
        """SFF = (safe + no_effect + DC*dangerous_detected) / total."""
        if not self.failure_modes:
            return 0.0
        for fm in self.failure_modes:
            if not isinstance(fm, FailureMode):
                raise ValueError("failure_modes must contain FailureMode entries")
        total = self.total_failure_rate
        if total == 0:
            return 0.0
        safe_sum = sum(fm.failure_rate_fit * fm.safe_failure_fraction for fm in self.failure_modes)
        return safe_sum / total

    @property
    def diagnostic_coverage(self) -> float:
        for fm in self.failure_modes:
            if not isinstance(fm, FailureMode):
                raise ValueError("failure_modes must contain FailureMode entries")
        dd = [fm for fm in self.failure_modes if fm.category == FailureCategory.DANGEROUS_DETECTED]
        if not dd:
            return 0.0
        for fm in dd:
            if not math.isfinite(float(fm.diagnostic_coverage)):
                raise ValueError("diagnostic_coverage entries must be finite values")
            if float(fm.diagnostic_coverage) < 0.0 or float(fm.diagnostic_coverage) > 1.0:
                raise ValueError("diagnostic_coverage entries must be in [0, 1]")
        weighted_sum = sum(fm.diagnostic_coverage * fm.failure_rate_fit for fm in dd)
        denominator = sum(fm.failure_rate_fit for fm in dd)
        if not math.isfinite(float(weighted_sum)) or weighted_sum < 0.0:
            raise ValueError("diagnostic_coverage weighted sum must be a finite non-negative value")
        if not math.isfinite(float(denominator)) or denominator <= 0.0:
            raise ValueError("diagnostic_coverage denominator must be a finite positive value")
        return weighted_sum / denominator

    @property
    def residual_risk_fit(self) -> float:
        """Dangerous-undetected failure rate (residual risk)."""
        for fm in self.failure_modes:
            if not isinstance(fm, FailureMode):
                raise ValueError("failure_modes must contain FailureMode entries")
        residual = sum(
            fm.failure_rate_fit * (1.0 - fm.safe_failure_fraction) for fm in self.failure_modes
        )
        if not math.isfinite(float(residual)) or residual < 0.0:
            raise ValueError("residual_risk_fit must be a finite non-negative value")
        return residual

    def sff_by_component(self) -> Dict[str, float]:
        """Per-component safe failure fraction."""
        components: Dict[str, List[FailureMode]] = {}
        for fm in self.failure_modes:
            if not isinstance(fm, FailureMode):
                raise ValueError("failure_modes must contain FailureMode entries")
            if not isinstance(fm.component, str) or not fm.component.strip():
                raise ValueError("failure modes must have non-empty component names")
            components.setdefault(fm.component, []).append(fm)
        result = {}
        for comp, fms in components.items():
            total = sum(f.failure_rate_fit for f in fms)
            safe = sum(f.failure_rate_fit * f.safe_failure_fraction for f in fms)
            if not math.isfinite(float(total)) or total < 0.0:
                raise ValueError("component failure-rate totals must be finite non-negative values")
            if not math.isfinite(float(safe)) or safe < 0.0:
                raise ValueError("component safe-failure totals must be finite non-negative values")
            result[comp] = safe / total if total > 0 else 0.0
        return result

    def max_achievable_sil(self) -> SILLevel:
        """Determine max achievable SIL from SFF and DC."""
        sff = self.safe_failure_fraction
        dc = self.diagnostic_coverage
        for value, name in ((sff, "safe_failure_fraction"), (dc, "diagnostic_coverage")):
            if not math.isfinite(float(value)) or float(value) < 0.0 or float(value) > 1.0:
                raise ValueError(f"{name} must be a finite value in [0, 1]")
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
        for fm in self.failure_modes:
            if not isinstance(fm, FailureMode):
                raise ValueError("failure_modes must contain FailureMode entries")
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
        for fm in sorted(self.failure_modes, key=lambda mode: mode.fm_id):
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

    def __post_init__(self) -> None:
        if not isinstance(self.prop_id, str) or not self.prop_id.strip():
            raise ValueError("prop_id must be a non-empty string")
        if not isinstance(self.module, str) or not self.module.strip():
            raise ValueError("module must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if not isinstance(self.property_type, str) or self.property_type not in {
            "assert",
            "cover",
            "assume",
        }:
            raise ValueError("property_type must be one of: assert, cover, assume")
        if not isinstance(self.status, str) or self.status not in {"proven", "failed", "unknown"}:
            raise ValueError("status must be one of: proven, failed, unknown")
        if not isinstance(self.engine, str) or not self.engine.strip():
            raise ValueError("engine must be a non-empty string")
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 0:
            raise ValueError("depth must be a non-negative integer")
        if not isinstance(self.sby_file, str):
            raise ValueError("sby_file must be a string")


@dataclass
class FormalProofCertificate:
    """Certificate packaging formal verification evidence."""

    properties: List[FormalProperty] = field(default_factory=list)
    generation_timestamp: str = ""
    tool_version: str = "SymbiYosys"
    certificate_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.generation_timestamp, str):
            raise ValueError("generation_timestamp must be a string")
        if not isinstance(self.tool_version, str) or not self.tool_version.strip():
            raise ValueError("tool_version must be a non-empty string")
        if not isinstance(self.certificate_hash, str):
            raise ValueError("certificate_hash must be a string")
        for prop in self.properties:
            if not isinstance(prop, FormalProperty):
                raise ValueError("properties must contain FormalProperty entries")

    def add_property(self, prop: FormalProperty) -> None:
        if not isinstance(prop, FormalProperty):
            raise ValueError("prop must be a FormalProperty")
        self.properties.append(prop)

    @property
    def proven_count(self) -> int:
        for prop in self.properties:
            if not isinstance(prop, FormalProperty):
                raise ValueError("properties must contain FormalProperty entries")
            if not isinstance(prop.status, str) or prop.status not in {
                "proven",
                "failed",
                "unknown",
            }:
                raise ValueError("properties statuses must be one of: proven, failed, unknown")
        return sum(1 for p in self.properties if p.status == "proven")

    @property
    def total_count(self) -> int:
        for prop in self.properties:
            if not isinstance(prop, FormalProperty):
                raise ValueError("properties must contain FormalProperty entries")
            if not isinstance(prop.prop_id, str) or not prop.prop_id.strip():
                raise ValueError("properties prop_id values must be non-empty strings")
        return len(self.properties)

    @property
    def pass_rate(self) -> float:
        return self.proven_count / self.total_count if self.total_count > 0 else 0.0

    def compute_hash(self) -> str:
        for prop in self.properties:
            if not isinstance(prop, FormalProperty):
                raise ValueError("properties must contain FormalProperty entries")
            if not isinstance(prop.module, str) or not prop.module.strip():
                raise ValueError("properties modules must be non-empty strings")
        prop_ids = [p.prop_id for p in self.properties]
        if len(prop_ids) != len(set(prop_ids)):
            raise ValueError("properties must not contain duplicate prop_id values")
        h = hashlib.sha256()
        for p in sorted(self.properties, key=lambda x: x.prop_id):
            h.update(f"{p.prop_id}:{p.status}:{p.module}".encode())
        self.certificate_hash = h.hexdigest()[:32]
        self.generation_timestamp = datetime.now().isoformat()
        return self.certificate_hash

    def generate_report(self) -> str:
        for prop in self.properties:
            if not isinstance(prop, FormalProperty):
                raise ValueError("properties must contain FormalProperty entries")
            if not isinstance(prop.prop_id, str) or not prop.prop_id.strip():
                raise ValueError("properties prop_id values must be non-empty strings")
            if not isinstance(prop.property_type, str) or prop.property_type not in {
                "assert",
                "cover",
                "assume",
            }:
                raise ValueError(
                    "properties property_type values must be one of: assert, cover, assume"
                )
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

    def __post_init__(self) -> None:
        if not isinstance(self.path_id, str) or not self.path_id.strip():
            raise ValueError("path_id must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if len(self.stages) != len(self.cycles_per_stage):
            raise ValueError("stages and cycles_per_stage must have the same length")
        if not self.stages:
            raise ValueError("stages must not be empty")
        for stage in self.stages:
            if not isinstance(stage, str) or not stage.strip():
                raise ValueError("stages must contain non-empty strings")
        for cycles in self.cycles_per_stage:
            if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 0:
                raise ValueError("cycles_per_stage must contain non-negative integers")

    @property
    def total_cycles(self) -> int:
        for cycles in self.cycles_per_stage:
            if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles < 0:
                raise ValueError("cycles_per_stage must contain non-negative integers")
        return sum(self.cycles_per_stage)

    def wcet_ns(self, clock_mhz: float) -> float:
        if isinstance(clock_mhz, bool) or not isinstance(clock_mhz, int | float):
            raise ValueError("clock_mhz must be a finite positive number")
        if not math.isfinite(float(clock_mhz)) or float(clock_mhz) <= 0.0:
            raise ValueError("clock_mhz must be a finite positive number")
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
        for value, field_name, minimum in (
            (bitstream_length, "bitstream_length", 1),
            (num_inputs, "num_inputs", 1),
            (num_neurons, "num_neurons", 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{field_name} must be an integer >= {minimum}")
        if not isinstance(has_stp, bool):
            raise ValueError("has_stp must be a boolean")
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
        if not isinstance(layers, list) or not layers:
            raise ValueError("layers must be a non-empty list")
        stages = []
        cycles = []
        for i, layer in enumerate(layers):
            if not isinstance(layer, dict):
                raise ValueError("each layer must be a dictionary")
            bs = layer.get("bitstream_length", 256)
            ni = layer.get("num_inputs", 8)
            nn = layer.get("num_neurons", 16)
            for value, field_name, minimum in (
                (bs, f"layers[{i}].bitstream_length", 1),
                (ni, f"layers[{i}].num_inputs", 1),
                (nn, f"layers[{i}].num_neurons", 1),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                    raise ValueError(f"{field_name} must be an integer >= {minimum}")
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

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or not self.item_id.strip():
            raise ValueError("item_id must be a non-empty string")
        if not isinstance(self.clause, str) or not self.clause.strip():
            raise ValueError("clause must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if not isinstance(self.evidence, str):
            raise ValueError("evidence must be a string")
        if not isinstance(self.status, str) or self.status not in {
            "compliant",
            "partial",
            "not_addressed",
        }:
            raise ValueError("status must be one of: compliant, partial, not_addressed")


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
        if not isinstance(standard, SafetyStandard):
            raise ValueError("standard must be a SafetyStandard")
        clause_map = {
            SafetyStandard.IEC_61508: cls.IEC_61508_CLAUSES,
            SafetyStandard.ISO_26262: cls.ISO_26262_CLAUSES,
            SafetyStandard.FDA_CLASS_III: cls.FDA_CLASS_III_CLAUSES,
            SafetyStandard.DO_254: cls.DO_254_CLAUSES,
            SafetyStandard.EN_50129: cls.EN_50129_CLAUSES,
        }
        clauses = clause_map.get(standard, [])
        for entry in clauses:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 3
                or not isinstance(entry[0], str)
                or not entry[0].strip()
                or not isinstance(entry[1], str)
                or not entry[1].strip()
                or not isinstance(entry[2], str)
                or not entry[2].strip()
            ):
                raise ValueError(
                    "clause definitions must contain non-empty (clause, description, evidence) tuples"
                )
        if len({clause for clause, _, _ in clauses}) != len(clauses):
            raise ValueError("clause definitions must not contain duplicates")
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

    def __post_init__(self) -> None:
        if not isinstance(self.standard, SafetyStandard):
            raise ValueError("standard must be a SafetyStandard")
        if not isinstance(self.sil_level, SILLevel):
            raise ValueError("sil_level must be a SILLevel")
        for field_name in (
            "traceability_report",
            "fmeda_report",
            "formal_cert_report",
            "wcet_report",
            "package_hash",
            "generated",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string")
        for item in self.checklist:
            if not isinstance(item, ChecklistItem):
                raise ValueError("checklist must contain ChecklistItem entries")
            if not isinstance(item.item_id, str) or not item.item_id.strip():
                raise ValueError("checklist item_id values must be non-empty strings")
            if not isinstance(item.clause, str) or not item.clause.strip():
                raise ValueError("checklist clauses must be non-empty strings")
            if not isinstance(item.status, str) or item.status not in {
                "compliant",
                "partial",
                "not_addressed",
            }:
                raise ValueError(
                    "checklist statuses must be one of: compliant, partial, not_addressed"
                )
            if not isinstance(item.clause, str) or not item.clause.strip():
                raise ValueError("checklist clauses must be non-empty strings")

    @property
    def checklist_coverage(self) -> float:
        if not self.checklist:
            return 0.0
        for item in self.checklist:
            if not isinstance(item, ChecklistItem):
                raise ValueError("checklist must contain ChecklistItem entries")
            if not isinstance(item.status, str) or item.status not in {
                "compliant",
                "partial",
                "not_addressed",
            }:
                raise ValueError(
                    "checklist statuses must be one of: compliant, partial, not_addressed"
                )
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
        if not isinstance(standard, SafetyStandard):
            raise ValueError("standard must be a SafetyStandard")
        if not isinstance(target_sil, SILLevel):
            raise ValueError("target_sil must be a SILLevel")
        if not isinstance(modules, list) or not modules:
            raise ValueError("modules must be a non-empty list")
        for module in modules:
            if not isinstance(module, str) or not module.strip():
                raise ValueError("modules must contain non-empty strings")
            if module != module.strip():
                raise ValueError("modules must not contain leading or trailing whitespace")
        if len(modules) != len(set(modules)):
            raise ValueError("modules must not contain duplicates")
        if not isinstance(formal_properties, list):
            raise ValueError("formal_properties must be a list")
        for prop in formal_properties:
            if not isinstance(prop, FormalProperty):
                raise ValueError("formal_properties must contain FormalProperty entries")
            if not isinstance(prop.prop_id, str) or not prop.prop_id.strip():
                raise ValueError("formal_properties prop_id values must be non-empty strings")
            if prop.prop_id != prop.prop_id.strip():
                raise ValueError(
                    "formal_properties prop_id values must not contain leading or trailing whitespace"
                )
            if not isinstance(prop.module, str) or not prop.module.strip():
                raise ValueError("formal_properties modules must be non-empty strings")
            if prop.module != prop.module.strip():
                raise ValueError(
                    "formal_properties modules must not contain leading or trailing whitespace"
                )
            if not isinstance(prop.status, str) or prop.status not in {
                "proven",
                "failed",
                "unknown",
            }:
                raise ValueError(
                    "formal_properties statuses must be one of: proven, failed, unknown"
                )
            if not isinstance(prop.property_type, str) or prop.property_type not in {
                "assert",
                "cover",
                "assume",
            }:
                raise ValueError(
                    "formal_properties property_type values must be one of: assert, cover, assume"
                )
        prop_ids = [prop.prop_id for prop in formal_properties]
        if len(prop_ids) != len(set(prop_ids)):
            raise ValueError("formal_properties must not contain duplicate prop_id values")
        if network_config is not None and not isinstance(network_config, dict):
            raise ValueError("network_config must be a dictionary when provided")
        if network_config is not None:
            allowed_keys = {"bitstream_length", "num_inputs", "num_neurons", "clock_mhz"}
            unknown_keys = [key for key in network_config if key not in allowed_keys]
            if unknown_keys:
                raise ValueError("network_config contains unsupported keys")
            for key in ("bitstream_length", "num_inputs", "num_neurons"):
                if key in network_config:
                    value = network_config[key]
                    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                        raise ValueError(f"network_config[{key}] must be an integer >= 1")
            if "clock_mhz" in network_config:
                clock = network_config["clock_mhz"]
                if isinstance(clock, bool) or not isinstance(clock, int | float):
                    raise ValueError("network_config[clock_mhz] must be a finite positive number")
                if not math.isfinite(float(clock)) or float(clock) <= 0.0:
                    raise ValueError("network_config[clock_mhz] must be a finite positive number")

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

    def __post_init__(self) -> None:
        if not isinstance(self.defence_id, str) or not self.defence_id.strip():
            raise ValueError("defence_id must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if not isinstance(self.category, str) or self.category not in {
            "separation",
            "diversity",
            "complexity",
            "assessment",
            "competence",
        }:
            raise ValueError(
                "category must be one of: separation, diversity, complexity, assessment, competence"
            )
        if isinstance(self.beta_reduction, bool) or not isinstance(
            self.beta_reduction, int | float
        ):
            raise ValueError("beta_reduction must be a finite non-negative value")
        if not math.isfinite(float(self.beta_reduction)) or float(self.beta_reduction) < 0.0:
            raise ValueError("beta_reduction must be a finite non-negative value")
        if float(self.beta_reduction) > 1.0:
            raise ValueError("beta_reduction must not exceed 1.0")
        if not isinstance(self.implemented, bool):
            raise ValueError("implemented must be a boolean")


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

    def __init__(self) -> None:
        for defence in self.DEFAULT_DEFENCES:
            if not isinstance(defence, CCFDefence):
                raise ValueError("DEFAULT_DEFENCES must contain CCFDefence entries")
        defence_ids = [defence.defence_id for defence in self.DEFAULT_DEFENCES]
        if len(defence_ids) != len(set(defence_ids)):
            raise ValueError("DEFAULT_DEFENCES must not contain duplicate defence_id values")
        self.defences: List[CCFDefence] = [
            CCFDefence(d.defence_id, d.description, d.category, d.beta_reduction, d.implemented)
            for d in self.DEFAULT_DEFENCES
        ]

    def mark_implemented(self, defence_id: str) -> bool:
        if not isinstance(defence_id, str) or not defence_id.strip():
            raise ValueError("defence_id must be a non-empty string")
        normalised_id = defence_id.strip()
        for d in self.defences:
            if not isinstance(d, CCFDefence):
                raise ValueError("defences must contain CCFDefence entries")
            if d.defence_id == normalised_id:
                d.implemented = True
                return True
        return False

    @property
    def beta_factor(self) -> float:
        """Resulting β-factor (start at 0.10, reduce by implemented defences)."""
        for defence in self.defences:
            if not isinstance(defence, CCFDefence):
                raise ValueError("defences must contain CCFDefence entries")
        base = 0.10
        reduction = 0.0
        for defence in self.defences:
            beta_reduction = float(defence.beta_reduction)
            if not math.isfinite(beta_reduction) or beta_reduction < 0.0 or beta_reduction > 1.0:
                raise ValueError("defence beta_reduction values must be finite values in [0, 1]")
            if defence.implemented:
                reduction += beta_reduction
        return max(0.005, base - reduction)

    @property
    def implemented_count(self) -> int:
        count = 0
        for defence in self.defences:
            if not isinstance(defence, CCFDefence):
                raise ValueError("defences must contain CCFDefence entries")
            if defence.implemented:
                count += 1
        return count

    def sil_compatible(self, target_sil: SILLevel) -> bool:
        """Check if β is low enough for the target SIL."""
        if not isinstance(target_sil, SILLevel):
            raise ValueError("target_sil must be a SILLevel")
        thresholds = {
            SILLevel.SIL_1: 0.10,
            SILLevel.SIL_2: 0.05,
            SILLevel.SIL_3: 0.02,
            SILLevel.SIL_4: 0.01,
        }
        return self.beta_factor <= thresholds[target_sil]


# ── Proof-of-Test Coverage (Gap 2) ──────────────────────────────────


class ProofTestCoverage:
    """Maps formal proof pass/fail to diagnostic coverage for SIL claims."""

    @staticmethod
    def coverage_from_proofs(properties: List[FormalProperty]) -> float:
        """Formal proof coverage = proven / total asserts."""
        if not isinstance(properties, list):
            raise ValueError("properties must be a list")
        for prop in properties:
            if not isinstance(prop, FormalProperty):
                raise ValueError("properties must contain FormalProperty entries")
            if not isinstance(prop.prop_id, str) or not prop.prop_id.strip():
                raise ValueError("properties prop_id values must be non-empty strings")
            if not isinstance(prop.status, str) or prop.status not in {
                "proven",
                "failed",
                "unknown",
            }:
                raise ValueError("properties statuses must be one of: proven, failed, unknown")
            if not isinstance(prop.property_type, str) or prop.property_type not in {
                "assert",
                "cover",
                "assume",
            }:
                raise ValueError(
                    "properties property_type values must be one of: assert, cover, assume"
                )
        asserts = [p for p in properties if p.property_type == "assert"]
        if not asserts:
            return 0.0
        proven = sum(1 for p in asserts if p.status == "proven")
        return proven / len(asserts)

    @staticmethod
    def dc_to_sil(dc: float) -> SILLevel:
        """Map diagnostic coverage to max SIL per IEC 61508 Table 3."""
        if isinstance(dc, bool) or not isinstance(dc, int | float):
            raise ValueError("dc must be a finite value in [0, 1]")
        value = float(dc)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("dc must be a finite value in [0, 1]")
        if value >= 0.99:
            return SILLevel.SIL_4
        if value >= 0.97:
            return SILLevel.SIL_3
        if value >= 0.90:
            return SILLevel.SIL_2
        if value >= 0.60:
            return SILLevel.SIL_1
        return SILLevel.SIL_1

    @staticmethod
    def uncovered_modules(properties: List[FormalProperty], all_modules: List[str]) -> List[str]:
        """Modules with no formal proofs."""
        if not isinstance(properties, list):
            raise ValueError("properties must be a list")
        for prop in properties:
            if not isinstance(prop, FormalProperty):
                raise ValueError("properties must contain FormalProperty entries")
            if not isinstance(prop.prop_id, str) or not prop.prop_id.strip():
                raise ValueError("properties prop_id values must be non-empty strings")
            if not isinstance(prop.module, str) or not prop.module.strip():
                raise ValueError("properties modules must be non-empty strings")
        if not isinstance(all_modules, list):
            raise ValueError("all_modules must be a list")
        for module in all_modules:
            if not isinstance(module, str) or not module.strip():
                raise ValueError("all_modules must contain non-empty strings")
            if module != module.strip():
                raise ValueError(
                    "all_modules entries must not contain leading or trailing whitespace"
                )
        covered = {p.module for p in properties}
        uncovered: list[str] = []
        seen: set[str] = set()
        for module in all_modules:
            if module not in covered and module not in seen:
                uncovered.append(module)
                seen.add(module)
        return uncovered


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

    def __post_init__(self) -> None:
        if isinstance(self.sff, bool) or not isinstance(self.sff, int | float):
            raise ValueError("sff must be a finite value in [0, 1]")
        if not math.isfinite(float(self.sff)) or float(self.sff) < 0.0 or float(self.sff) > 1.0:
            raise ValueError("sff must be a finite value in [0, 1]")
        if not isinstance(self.target_sil, SILLevel):
            raise ValueError("target_sil must be a SILLevel")

    @property
    def required_hft(self) -> HFTLevel:
        """Determine required HFT from SFF and target SIL."""
        if not isinstance(self.target_sil, SILLevel):
            raise ValueError("target_sil must be a SILLevel")
        if not math.isfinite(float(self.sff)) or float(self.sff) < 0.0 or float(self.sff) > 1.0:
            raise ValueError("sff must be a finite value in [0, 1]")
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

    def __post_init__(self) -> None:
        if not isinstance(self.change_id, str) or not self.change_id.strip():
            raise ValueError("change_id must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if not isinstance(self.risk_level, str) or self.risk_level not in {"low", "medium", "high"}:
            raise ValueError("risk_level must be one of: low, medium, high")
        if not isinstance(self.re_verification_needed, bool):
            raise ValueError("re_verification_needed must be a boolean")
        if not isinstance(self.affected_modules, list):
            raise ValueError("affected_modules must be a list")
        if not isinstance(self.affected_reqs, list):
            raise ValueError("affected_reqs must be a list")

        for module in self.affected_modules:
            if not isinstance(module, str) or not module.strip():
                raise ValueError("affected_modules must contain non-empty strings")
        for req in self.affected_reqs:
            if not isinstance(req, str) or not req.strip():
                raise ValueError("affected_reqs must contain non-empty strings")


class ChangeImpactTracker:
    """Tracks changes and their impact on certification artifacts."""

    def __init__(self) -> None:
        self.changes: List[ChangeRecord] = []

    def add_change(self, change: ChangeRecord) -> None:
        if not isinstance(change, ChangeRecord):
            raise ValueError("change must be a ChangeRecord")
        if any(existing.change_id == change.change_id for existing in self.changes):
            raise ValueError("change_id values must be unique")
        if change.risk_level in ("medium", "high"):
            change.re_verification_needed = True
        self.changes.append(change)

    def affected_requirements(self) -> List[str]:
        reqs: set[str] = set()
        for c in self.changes:
            if not isinstance(c, ChangeRecord):
                raise ValueError("changes must contain ChangeRecord entries")
            if not isinstance(c.change_id, str) or not c.change_id.strip():
                raise ValueError("change_id values must be non-empty strings")
            if not isinstance(c.affected_reqs, list):
                raise ValueError("affected_reqs must be a list")
            for req in c.affected_reqs:
                if not isinstance(req, str) or not req.strip():
                    raise ValueError("affected_reqs must contain non-empty strings")
            reqs.update(c.affected_reqs)
        return sorted(reqs)

    @property
    def high_risk_count(self) -> int:
        count = 0
        for change in self.changes:
            if not isinstance(change, ChangeRecord):
                raise ValueError("changes must contain ChangeRecord entries")
            if not isinstance(change.risk_level, str) or change.risk_level not in {
                "low",
                "medium",
                "high",
            }:
                raise ValueError("changes risk_level values must be one of: low, medium, high")
            if change.risk_level == "high":
                count += 1
        return count

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
        if not isinstance(product_name, str) or not product_name.strip():
            raise ValueError("product_name must be a non-empty string")
        if not isinstance(sil_level, SILLevel):
            raise ValueError("sil_level must be a SILLevel")
        if not isinstance(modules, list) or not modules:
            raise ValueError("modules must be a non-empty list")
        for module in modules:
            if not isinstance(module, str) or not module.strip():
                raise ValueError("modules must contain non-empty strings")
            if module != module.strip():
                raise ValueError("modules must not contain leading or trailing whitespace")
        if len(modules) != len(set(modules)):
            raise ValueError("modules must not contain duplicates")
        if isinstance(wcet_ns, bool) or not isinstance(wcet_ns, int | float):
            raise ValueError("wcet_ns must be a finite non-negative value")
        if not math.isfinite(float(wcet_ns)) or float(wcet_ns) < 0.0:
            raise ValueError("wcet_ns must be a finite non-negative value")
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

    def __post_init__(self) -> None:
        if not isinstance(self.sw_class, SWClass):
            raise ValueError("sw_class must be a SWClass")
        if not isinstance(self.hazard_description, str):
            raise ValueError("hazard_description must be a string")
        if not isinstance(self.risk_control_measures, list):
            raise ValueError("risk_control_measures must be a list")
        for measure in self.risk_control_measures:
            if not isinstance(measure, str) or not measure.strip():
                raise ValueError("risk_control_measures must contain non-empty strings")

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

    def __post_init__(self) -> None:
        if isinstance(self.total_fit, bool) or not isinstance(self.total_fit, int | float):
            raise ValueError("total_fit must be a finite non-negative number")
        if not math.isfinite(float(self.total_fit)) or float(self.total_fit) < 0.0:
            raise ValueError("total_fit must be a finite non-negative number")
        if isinstance(self.dangerous_undetected_fit, bool) or not isinstance(
            self.dangerous_undetected_fit, int | float
        ):
            raise ValueError("dangerous_undetected_fit must be a finite non-negative number")
        if (
            not math.isfinite(float(self.dangerous_undetected_fit))
            or float(self.dangerous_undetected_fit) < 0.0
        ):
            raise ValueError("dangerous_undetected_fit must be a finite non-negative number")
        if float(self.dangerous_undetected_fit) > float(self.total_fit):
            raise ValueError("dangerous_undetected_fit cannot exceed total_fit")

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
        if not math.isfinite(float(pfh)) or float(pfh) < 0.0:
            raise ValueError("pfh_d must be a finite non-negative number")
        if pfh <= 1e-8:
            return SILLevel.SIL_4
        if pfh <= 1e-7:
            return SILLevel.SIL_3
        if pfh <= 1e-6:
            return SILLevel.SIL_2
        return SILLevel.SIL_1

    @staticmethod
    def from_fmeda(fmeda: FMEDA) -> ReliabilityMetrics:
        if not isinstance(fmeda, FMEDA):
            raise ValueError("fmeda must be an FMEDA")
        return ReliabilityMetrics(
            total_fit=fmeda.total_failure_rate,
            dangerous_undetected_fit=fmeda.residual_risk_fit,
        )


# ── Evidence Bag Manifest (Gap 8) ───────────────────────────────────


@dataclass
class EvidenceItem:
    """One item in the evidence bag."""

    filename: str
    category: str
    description: str
    sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.filename, str) or not self.filename.strip():
            raise ValueError("filename must be a non-empty string")
        if not isinstance(self.category, str) or self.category not in {
            "formal",
            "test",
            "analysis",
            "design",
            "report",
            "hil",
            "hardware-in-loop",
            "hardware_in_loop",
            "security",
            "timing",
            "latency",
            "formal_timing",
        }:
            raise ValueError(
                "category must be one of: formal, test, analysis, design, report, "
                "hil, hardware-in-loop, hardware_in_loop, security, timing, latency, formal_timing"
            )
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if not isinstance(self.sha256, str):
            raise ValueError("sha256 must be a string")
        if self.sha256:
            if len(self.sha256) != 64 or any(
                c not in "0123456789abcdefABCDEF" for c in self.sha256
            ):
                raise ValueError("sha256 must be a 64-character hexadecimal digest when provided")


class EvidenceBag:
    """Manifest of all certification evidence artifacts."""

    def __init__(self) -> None:
        self.items: List[EvidenceItem] = []

    def add(self, item: EvidenceItem) -> None:
        if not isinstance(item, EvidenceItem):
            raise ValueError("item must be an EvidenceItem")
        if any(existing.filename == item.filename for existing in self.items):
            raise ValueError("evidence filenames must be unique")
        self.items.append(item)

    def add_from_package(self, pkg: CertificationPackage) -> None:
        if not isinstance(pkg, CertificationPackage):
            raise ValueError("pkg must be a CertificationPackage")
        for item in pkg.checklist:
            if not isinstance(item, ChecklistItem):
                raise ValueError("pkg.checklist must contain ChecklistItem entries")
        self.add(EvidenceItem("traceability_matrix.md", "report", "Requirement traceability"))
        self.add(EvidenceItem("fmeda_report.md", "analysis", "FMEDA analysis"))
        self.add(EvidenceItem("formal_proof_cert.md", "formal", "Formal proof certificate"))
        self.add(EvidenceItem("wcet_analysis.md", "analysis", "WCET analysis"))
        self.add(EvidenceItem("compliance_checklist.md", "report", "Compliance checklist"))

    @property
    def file_count(self) -> int:
        for item in self.items:
            if not isinstance(item, EvidenceItem):
                raise ValueError("items must contain EvidenceItem entries")
        return len(self.items)

    def manifest(self) -> str:
        for item in self.items:
            if not isinstance(item, EvidenceItem):
                raise ValueError("items must contain EvidenceItem entries")
        lines = ["# Evidence Bag Manifest", f"Items: {self.file_count}", ""]
        for item in self.items:
            lines.append(f"- [{item.category}] {item.filename}: {item.description}")
        return "\n".join(lines)

    def compute_hashes(self) -> str:
        for item in self.items:
            if not isinstance(item, EvidenceItem):
                raise ValueError("items must contain EvidenceItem entries")
        filenames = [item.filename for item in self.items]
        if len(filenames) != len(set(filenames)):
            raise ValueError("evidence filenames must be unique")
        h = hashlib.sha256()
        for item in sorted(self.items, key=lambda x: x.filename):
            h.update(f"{item.filename}:{item.category}:{item.sha256}".encode())
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
        if not isinstance(standard, str) or not standard.strip():
            raise ValueError("standard must be a non-empty string")
        if not isinstance(clause, str) or not clause.strip():
            raise ValueError("clause must be a non-empty string")
        mappings = CROSS_MAP.get((standard.strip(), clause.strip()), [])
        for mapping in mappings:
            if (
                not isinstance(mapping, tuple)
                or len(mapping) != 2
                or not isinstance(mapping[0], str)
                or not mapping[0].strip()
                or not isinstance(mapping[1], str)
                or not mapping[1].strip()
            ):
                raise ValueError(
                    "cross-standard mappings must contain non-empty (standard, clause) tuples"
                )
        return mappings

    @staticmethod
    def coverage_overlap(checklist_a: List[ChecklistItem], checklist_b: List[ChecklistItem]) -> int:
        """Count shared compliance coverage between two checklists."""
        if not isinstance(checklist_a, list) or not isinstance(checklist_b, list):
            raise ValueError("checklist_a and checklist_b must be lists")
        for item in checklist_a:
            if not isinstance(item, ChecklistItem):
                raise ValueError("checklist_a must contain ChecklistItem entries")
            if not isinstance(item.clause, str) or not item.clause.strip():
                raise ValueError("checklist_a clauses must be non-empty strings")
            if not isinstance(item.status, str) or item.status not in {
                "compliant",
                "partial",
                "not_addressed",
            }:
                raise ValueError(
                    "checklist_a statuses must be one of: compliant, partial, not_addressed"
                )
        for item in checklist_b:
            if not isinstance(item, ChecklistItem):
                raise ValueError("checklist_b must contain ChecklistItem entries")
            if not isinstance(item.clause, str) or not item.clause.strip():
                raise ValueError("checklist_b clauses must be non-empty strings")
            if not isinstance(item.status, str) or item.status not in {
                "compliant",
                "partial",
                "not_addressed",
            }:
                raise ValueError(
                    "checklist_b statuses must be one of: compliant, partial, not_addressed"
                )
        addressed_a = {i.clause for i in checklist_a if i.status != "not_addressed"}
        addressed_b = {i.clause for i in checklist_b if i.status != "not_addressed"}
        shared_pairs: set[tuple[str, str]] = set()
        addressed_items_a = [i for i in checklist_a if i.clause in addressed_a]
        addressed_items_b = [i for i in checklist_b if i.clause in addressed_b]
        for item in addressed_items_a:
            if "_" not in item.item_id:
                raise ValueError("checklist_a item_id must contain standard and clause separator")
        for item in addressed_items_b:
            if "_" not in item.item_id:
                raise ValueError("checklist_b item_id must contain standard and clause separator")
        for std_a, clause_a in [
            (item.item_id.rsplit("_", 1)[0], item.clause) for item in addressed_items_a
        ]:
            for mapping in CROSS_MAP.get((std_a, clause_a), []):
                if mapping[1] in addressed_b:
                    shared_pairs.add((clause_a, mapping[1]))
        return len(shared_pairs)


# ── Formal Property Gap Detector (Gap 10) ───────────────────────────


@dataclass
class PropertyGap:
    """One module with insufficient formal coverage."""

    module: str
    total_properties: int
    proven_properties: int
    missing_types: List[str]  # property types not covered

    def __post_init__(self) -> None:
        if not isinstance(self.module, str) or not self.module.strip():
            raise ValueError("module must be a non-empty string")
        if isinstance(self.total_properties, bool) or not isinstance(self.total_properties, int):
            raise ValueError("total_properties must be a non-negative integer")
        if self.total_properties < 0:
            raise ValueError("total_properties must be a non-negative integer")
        if isinstance(self.proven_properties, bool) or not isinstance(self.proven_properties, int):
            raise ValueError("proven_properties must be a non-negative integer")
        if self.proven_properties < 0:
            raise ValueError("proven_properties must be a non-negative integer")
        if self.proven_properties > self.total_properties:
            raise ValueError("proven_properties cannot exceed total_properties")
        for item in self.missing_types:
            if not isinstance(item, str) or not item.strip():
                raise ValueError("missing_types must contain non-empty strings")

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
        if not isinstance(properties, list):
            raise ValueError("properties must be a list")
        for prop in properties:
            if not isinstance(prop, FormalProperty):
                raise ValueError("properties must contain FormalProperty entries")
            if not isinstance(prop.prop_id, str) or not prop.prop_id.strip():
                raise ValueError("properties prop_id values must be non-empty strings")
            if not isinstance(prop.property_type, str) or prop.property_type not in {
                "assert",
                "cover",
                "assume",
            }:
                raise ValueError(
                    "properties property_type values must be one of: assert, cover, assume"
                )
            if not isinstance(prop.status, str) or prop.status not in {
                "proven",
                "failed",
                "unknown",
            }:
                raise ValueError("properties statuses must be one of: proven, failed, unknown")
            if not isinstance(prop.module, str) or not prop.module.strip():
                raise ValueError("properties modules must be non-empty strings")
        if not isinstance(required_modules, list):
            raise ValueError("required_modules must be a list")
        for module in required_modules:
            if not isinstance(module, str) or not module.strip():
                raise ValueError("required_modules must contain non-empty strings")
            if module != module.strip():
                raise ValueError(
                    "required_modules entries must not contain leading or trailing whitespace"
                )
        by_module: Dict[str, List[FormalProperty]] = {}
        for p in properties:
            by_module.setdefault(p.module, []).append(p)

        gaps = []
        seen_required: set[str] = set()
        for mod in required_modules:
            if mod in seen_required:
                continue
            seen_required.add(mod)
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
