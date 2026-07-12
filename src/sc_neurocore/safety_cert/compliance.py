# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Automated Safety & Regulatory Certification Generator

"""Fail-closed checklist templates and non-normative cross-standard mapping."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

from sc_neurocore.safety_cert.standards import SILLevel, SafetyStandard


@dataclass
class ChecklistItem:
    """One item in a compliance checklist."""

    item_id: str
    clause: str
    description: str
    evidence: str = ""
    status: str = "not_addressed"  # "compliant" | "partial" | "not_addressed"

    def __post_init__(self) -> None:
        """Validate one checklist row and its evidence state."""
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
        if self.status != "not_addressed" and not self.evidence.strip():
            raise ValueError("addressed checklist items require non-empty evidence")


class ComplianceChecklist:
    """Generate fail-closed checklist skeletons from curated clause labels.

    Clause descriptions and suggested artifact locations are navigation aids,
    not normative text. Users remain responsible for licensed standards,
    applicability analysis, evidence review, and conformity assessment.
    """

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
    def generate(
        cls,
        standard: SafetyStandard,
        *,
        evidence: Mapping[str, str] | None = None,
    ) -> List[ChecklistItem]:
        """Create a checklist, marking only caller-supplied evidence partial.

        The mapping is keyed by clause. Merely having a template or repository
        path never changes an item from not-addressed.
        """
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
        if evidence is not None and not isinstance(evidence, Mapping):
            raise ValueError("evidence must be a clause-to-reference mapping when provided")
        evidence_by_clause = dict(evidence or {})
        for clause in evidence_by_clause:
            if not isinstance(clause, str) or not clause.strip():
                raise ValueError("evidence clause keys must be non-empty strings")
        known_clauses = {clause for clause, _, _ in clauses}
        unknown_clauses = sorted(set(evidence_by_clause) - known_clauses)
        if unknown_clauses:
            raise ValueError("evidence contains clauses outside the selected checklist")
        for clause, reference in evidence_by_clause.items():
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError("evidence references must be non-empty strings")
        items = []
        for clause, desc, _suggested_location in clauses:
            reference = evidence_by_clause.get(clause, "")
            items.append(
                ChecklistItem(
                    item_id=f"{standard.value}_{clause}",
                    clause=clause,
                    description=desc,
                    evidence=reference,
                    status="partial" if reference else "not_addressed",
                )
            )
        return items


class SWClass(Enum):
    """IEC 62304 software safety-class labels."""

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
        """Validate the software-class assessment inputs."""
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
        """Return whether the legacy class screen calls for unit-test evidence."""
        return self.sw_class in (SWClass.CLASS_B, SWClass.CLASS_C)

    @property
    def requires_architectural_design(self) -> bool:
        """Return whether the legacy class screen calls for architecture evidence."""
        return self.sw_class == SWClass.CLASS_C

    @staticmethod
    def from_sil(sil: SILLevel) -> IEC62304Assessment:
        """Return the legacy, non-normative SIL-to-software-class crosswalk."""
        if not isinstance(sil, SILLevel):
            raise ValueError("sil must be a SILLevel")
        mapping = {
            SILLevel.SIL_1: SWClass.CLASS_A,
            SILLevel.SIL_2: SWClass.CLASS_B,
            SILLevel.SIL_3: SWClass.CLASS_C,
            SILLevel.SIL_4: SWClass.CLASS_C,
        }
        return IEC62304Assessment(sw_class=mapping[sil])


CROSS_MAP = {
    ("IEC 61508", "7.4.2"): [("ISO 26262", "6.7.4"), ("DO-254", "6.0")],
    ("IEC 61508", "7.4.3"): [("ISO 26262", "6.4.1")],
    ("IEC 61508", "7.6.2"): [("ISO 26262", "6.4.2"), ("FDA Class III", "820.72")],
    ("IEC 61508", "7.9.2"): [("ISO 26262", "9.4.1"), ("FDA Class III", "820.30")],
    ("ISO 26262", "8.4.3"): [("FDA Class III", "820.30(g)")],
}
"""Curated, non-normative navigation links between clause labels."""


class CrossStandardMapper:
    """Navigate curated clause relationships without asserting equivalence."""

    @staticmethod
    def equivalent_clauses(standard: str, clause: str) -> List[Tuple[str, str]]:
        """Return a copy of curated related-clause labels."""
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
        return list(mappings)

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


__all__ = [
    "CROSS_MAP",
    "ChecklistItem",
    "ComplianceChecklist",
    "SWClass",
    "IEC62304Assessment",
    "CrossStandardMapper",
]
