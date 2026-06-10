# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reporting facade

"""Advanced compilation reporting and analysis facade."""

from __future__ import annotations

from .compilation_report import (
    generate_compilation_report,
)
from .compilation_summary import (
    generate_compilation_summary,
)
from .model_complexity import (
    ModelComplexity,
    classify_model_complexity,
)
from .pareto_explorer import (
    ParetoPoint,
    explore_pareto,
)
from .portability_scorer import (
    PortabilityScore,
    score_portability,
)
from .provenance_chain import (
    ProvenanceRecord,
    format_provenance_json,
    generate_provenance_chain,
)
from .target_comparison import (
    TargetComparison,
    compare_targets,
    format_comparison_report,
)

__all__ = [
    "ModelComplexity",
    "ParetoPoint",
    "PortabilityScore",
    "ProvenanceRecord",
    "TargetComparison",
    "classify_model_complexity",
    "compare_targets",
    "explore_pareto",
    "format_comparison_report",
    "format_provenance_json",
    "generate_compilation_report",
    "generate_compilation_summary",
    "generate_provenance_chain",
    "score_portability",
]
