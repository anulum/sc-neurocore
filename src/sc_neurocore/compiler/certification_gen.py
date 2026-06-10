# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Certification evidence generation

"""Certification evidence generation utilities for safety-critical deployment.

Generates XML traceability matrices for standards like DO-254 (avionics),
IEC 61508 (industrial), or ISO 26262 (automotive).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class CertificationItem:
    """A single certification evidence item.

    Attributes
    ----------
    req_id : str
        Requirement identifier (e.g. ``"REQ-001"``).
    description : str
        Requirement description.
    design_ref : str
        Design artifact (e.g. Verilog module name).
    verification_ref : str
        Verification artifact (e.g. SVA property, Cocotb test).
    status : str
        ``"PASS"``, ``"FAIL"``, or ``"UNTESTED"``.
    """

    req_id: str
    description: str
    design_ref: str
    verification_ref: str
    status: Literal["PASS", "FAIL", "UNTESTED"] = "UNTESTED"


def generate_certification_evidence(
    module_name: str,
    items: list[CertificationItem],
    *,
    standard: Literal["do254", "iec61508", "iso26262"] = "do254",
    dal_level: str = "DAL-C",
) -> str:
    """Generate XML traceability matrix for safety certification.

    Produces a certification evidence document linking requirements to
    design and verification artifacts in the format required by DO-254
    (avionics), IEC 61508 (industrial), or ISO 26262 (automotive).

    Parameters
    ----------
    module_name : str
        Design module under certification.
    items : list[CertificationItem]
        Requirement-to-evidence mapping.
    standard : str
        ``"do254"``, ``"iec61508"``, or ``"iso26262"``.
    dal_level : str
        Design Assurance Level or SIL/ASIL level.

    Returns
    -------
    str
        XML certification evidence document.
    """
    std_label = {
        "do254": "RTCA DO-254",
        "iec61508": "IEC 61508",
        "iso26262": "ISO 26262",
    }[standard]

    pass_count = sum(1 for i in items if i.status == "PASS")
    fail_count = sum(1 for i in items if i.status == "FAIL")
    total = len(items)
    pct = (pass_count / total * 100) if total else 0.0

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f"<!-- SC-NeuroCore Certification Evidence: {module_name} -->",
        "<certification_evidence>",
        f"  <module>{module_name}</module>",
        f"  <standard>{std_label}</standard>",
        f"  <level>{dal_level}</level>",
        f'  <summary total="{total}" passed="{pass_count}" '
        f'failed="{fail_count}" coverage="{pct:.1f}"/>',
        "  <traceability_matrix>",
    ]

    for item in items:
        lines.extend(
            [
                f'    <requirement id="{item.req_id}" status="{item.status}">',
                f"      <description>{item.description}</description>",
                f"      <design_ref>{item.design_ref}</design_ref>",
                f"      <verification_ref>{item.verification_ref}</verification_ref>",
                "    </requirement>",
            ]
        )

    lines.extend(
        [
            "  </traceability_matrix>",
            "</certification_evidence>",
            "",
        ]
    )

    return "\n".join(lines)
