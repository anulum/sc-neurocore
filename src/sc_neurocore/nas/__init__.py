# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hardware-aware SNN neural architecture search

"""Hardware-aware SNN NAS: search {neuron, width, delays, L} under FPGA budgets."""

from .search import NASResult, nas
from .search_space import Architecture, SearchSpace
from .surrogate_bridge import (
    NASPolicyEvaluation,
    NASPolicyLayer,
    NASPolicyPlan,
    apply_surrogate_policy,
    build_nas_policy_plan,
    candidate_layer_profiles,
    evaluate_candidate_with_surrogate,
    optimise_candidate_policy,
)

__all__ = [
    "Architecture",
    "SearchSpace",
    "nas",
    "NASResult",
    "NASPolicyLayer",
    "NASPolicyPlan",
    "NASPolicyEvaluation",
    "apply_surrogate_policy",
    "build_nas_policy_plan",
    "candidate_layer_profiles",
    "evaluate_candidate_with_surrogate",
    "optimise_candidate_policy",
]
