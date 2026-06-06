# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive precision facade

"""Per-layer adaptive bitstream length and per-synapse bit width facade."""

from __future__ import annotations

from .auto_tune import (
    auto_tune_synapse_precisions,
    precision_plan_manifest,
)
from .formal_evidence import (
    write_precision_formal_evidence_bundle,
)
from .layer_precision import (
    LayerPrecision,
)
from .length_planner import (
    assign_lengths,
)
from .sensitivity_analysis import (
    analyze_sensitivity,
)
from .synapse_planner import (
    assign_synapse_precisions,
)
from .synapse_precision import (
    SynapsePrecision,
)

__all__ = [
    "LayerPrecision",
    "SynapsePrecision",
    "analyze_sensitivity",
    "assign_lengths",
    "assign_synapse_precisions",
    "auto_tune_synapse_precisions",
    "precision_plan_manifest",
    "write_precision_formal_evidence_bundle",
]
