# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compliance matrix generator

"""Safety compliance matrix generation for certification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ComplianceEntry:
    """Single compliance requirement mapping.

    Attributes
    ----------
    req_id : str
        Requirement identifier.
    standard : str
        Safety standard name.
    description : str
        Requirement description.
    verification : str
        How it is verified.
    status : str
        ``"covered"``, ``"partial"``, or ``"gap"``.
    artefact : str
        File or test that provides evidence.
    """

    req_id: str
    standard: str
    description: str
    verification: str
    status: str
    artefact: str


def generate_compliance_matrix(
    module_name: str,
    *,
    standards: list[str] | None = None,
    has_tmr: bool = False,
    has_checksum: bool = False,
    has_sva: bool = False,
    has_provenance: bool = False,
) -> list[ComplianceEntry]:
    """Generate safety compliance matrix for certification.

    Maps DO-254 / IEC 61508 / ISO 26262 requirements to SC-NeuroCore
    verification artefacts.

    Parameters
    ----------
    module_name : str
        Module under certification.
    standards : list[str], optional
        Standards to cover. Default: all three.
    has_tmr : bool
        TMR wrapper is present.
    has_checksum : bool
        Model checksum is embedded.
    has_sva : bool
        SVA assertions are generated.
    has_provenance : bool
        Provenance chain exists.

    Returns
    -------
    list[ComplianceEntry]
        Compliance matrix entries.
    """
    if standards is None:
        standards = ["DO-254", "IEC 61508", "ISO 26262"]

    entries = []

    if "DO-254" in standards:
        entries.extend(
            [
                ComplianceEntry(
                    "DO254-1",
                    "DO-254",
                    "Design assurance level assignment",
                    "Compilation summary report",
                    "covered",
                    f"{module_name}_compilation_summary.md",
                ),
                ComplianceEntry(
                    "DO254-2",
                    "DO-254",
                    "Requirement traceability",
                    "Provenance chain",
                    "covered" if has_provenance else "gap",
                    f"{module_name}_provenance.json",
                ),
                ComplianceEntry(
                    "DO254-3",
                    "DO-254",
                    "SEU mitigation",
                    "TMR wrapper with majority voter",
                    "covered" if has_tmr else "gap",
                    f"{module_name}_tmr.v",
                ),
                ComplianceEntry(
                    "DO254-4",
                    "DO-254",
                    "Formal verification",
                    "SVA assertions + SymbiYosys proof",
                    "covered" if has_sva else "partial",
                    f"{module_name}_sva.sv",
                ),
                ComplianceEntry(
                    "DO254-5",
                    "DO-254",
                    "Configuration control",
                    "Model checksum (SHA-256)",
                    "covered" if has_checksum else "gap",
                    f"{module_name}_checksum.v",
                ),
            ]
        )

    if "IEC 61508" in standards:
        entries.extend(
            [
                ComplianceEntry(
                    "IEC61508-1",
                    "IEC 61508",
                    "SIL determination",
                    "Compilation summary + resource estimation",
                    "covered",
                    f"{module_name}_compilation_summary.md",
                ),
                ComplianceEntry(
                    "IEC61508-2",
                    "IEC 61508",
                    "Diagnostic coverage",
                    "TMR + checksum",
                    "covered" if (has_tmr and has_checksum) else "partial",
                    f"{module_name}_tmr.v",
                ),
            ]
        )

    if "ISO 26262" in standards:
        entries.extend(
            [
                ComplianceEntry(
                    "ISO26262-1",
                    "ISO 26262",
                    "ASIL decomposition",
                    "Multi-target comparison report",
                    "covered",
                    f"{module_name}_comparison.md",
                ),
                ComplianceEntry(
                    "ISO26262-2",
                    "ISO 26262",
                    "Fault injection",
                    "Weight noise injection + TMR",
                    "covered" if has_tmr else "partial",
                    f"{module_name}_noise_test.py",
                ),
            ]
        )

    return entries


def format_compliance_report(
    entries: list[ComplianceEntry],
) -> str:
    """Format compliance matrix as markdown."""
    lines = [
        "# SC-NeuroCore Safety Compliance Matrix",
        "",
        "| ID | Standard | Requirement | Verification | Status | Artefact |",
        "|-----|----------|-------------|-------------|--------|----------|",
    ]
    for e in entries:
        status_icon = {"covered": "✅", "partial": "⚠️", "gap": "❌"}.get(e.status, "?")
        lines.append(
            f"| {e.req_id} | {e.standard} | {e.description} "
            f"| {e.verification} | {status_icon} {e.status} | {e.artefact} |"
        )
    return "\n".join(lines)
