# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive precision planner contracts

"""Focused adaptive precision planner contracts."""

from __future__ import annotations


from sc_neurocore.compiler.adaptive_precision import (
    precision_plan_manifest,
)


def test_precision_plan_manifest_handles_empty_assignment_lists() -> None:
    """The public manifest API reports zero costs for an empty precision plan."""
    manifest = precision_plan_manifest([])

    assert manifest["num_synapses"] == 0
    assert manifest["cost_summary"]["estimated_lut_cost"] == 0.0
