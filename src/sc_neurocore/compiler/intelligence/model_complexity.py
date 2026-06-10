# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Model complexity classifier

"""Model compute-profile classification and platform recommendation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelComplexity:
    """Model compute-profile classification.

    Attributes
    ----------
    classification : str
        ``"compute_bound"``, ``"memory_bound"``, or ``"comm_bound"``.
    compute_ops : int
        Total arithmetic operations.
    memory_vars : int
        State variables.
    comm_ratio : float
        Inter-variable coupling ratio.
    recommended_paradigm : str
        Best platform class.
    """

    classification: str
    compute_ops: int
    memory_vars: int
    comm_ratio: float
    recommended_paradigm: str


def classify_model_complexity(
    equations: dict[str, str],
) -> ModelComplexity:
    """Classify a model's compute profile."""
    num_vars = len(equations)
    total_ops = sum(
        e.count("+") + e.count("-") + e.count("*") + e.count("/") for e in equations.values()
    )

    # Communication: count cross-variable references
    cross_refs = 0
    for sv, expr in equations.items():
        for other_sv in equations:
            if other_sv != sv and other_sv in expr:
                cross_refs += 1

    comm_ratio = cross_refs / max(1, num_vars)

    if total_ops / max(1, num_vars) > 4:
        cls = "compute_bound"
        paradigm = "fpga"
    elif num_vars > 4 and total_ops / max(1, num_vars) <= 2:
        cls = "memory_bound"
        paradigm = "in_memory"
    elif comm_ratio > 1.5:
        cls = "comm_bound"
        paradigm = "cgra"
    else:
        cls = "compute_bound"
        paradigm = "fpga"

    return ModelComplexity(
        classification=cls,
        compute_ops=total_ops,
        memory_vars=num_vars,
        comm_ratio=round(comm_ratio, 2),
        recommended_paradigm=paradigm,
    )
