# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio platform evidence classification re-export

"""Platform re-export for the shared Studio evidence classification contract."""

from __future__ import annotations

from sc_neurocore.studio.evidence_classification import (
    STUDIO_EVIDENCE_CLASSIFICATIONS,
    STUDIO_EVIDENCE_TERMINAL_STATUSES,
    StudioEvidenceClassification,
    StudioEvidenceStatus,
    validate_studio_evidence_classification,
    validate_studio_evidence_status,
)


__all__ = [
    "STUDIO_EVIDENCE_CLASSIFICATIONS",
    "STUDIO_EVIDENCE_TERMINAL_STATUSES",
    "StudioEvidenceClassification",
    "StudioEvidenceStatus",
    "validate_studio_evidence_classification",
    "validate_studio_evidence_status",
]
