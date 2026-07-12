# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Automated Safety & Regulatory Certification Generator

"""Safety-standard identifiers and compatibility crosswalks."""

from __future__ import annotations

from enum import Enum


class SafetyStandard(Enum):
    """Standards for which the library can organise user-supplied evidence."""

    IEC_61508 = "IEC 61508"
    ISO_26262 = "ISO 26262"
    FDA_CLASS_III = "FDA Class III"
    DO_254 = "DO-254"
    EN_50129 = "EN 50129"


class SILLevel(Enum):
    """Safety Integrity Level labels used in reports and screening utilities."""

    SIL_1 = 1
    SIL_2 = 2
    SIL_3 = 3
    SIL_4 = 4


class ASILLevel(Enum):
    """Automotive Safety Integrity Level labels used by a legacy crosswalk."""

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
"""Non-normative label crosswalk retained for API compatibility.

This mapping is not an equivalence assessment and must not be used to infer
compliance with either IEC 61508 or ISO 26262.
"""

__all__ = [
    "SIL_TO_ASIL",
    "SafetyStandard",
    "SILLevel",
    "ASILLevel",
]
