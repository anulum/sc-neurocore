# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fault tree generator

"""Fault Tree Analysis for safety certification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class FaultTree:
    """Fault Tree Analysis for safety certification.

    Attributes
    ----------
    top_event : str
        Top-level failure event.
    gates : list[dict]
        Logic gates (AND/OR).
    basic_events : list[dict]
        Leaf failure events with rates.
    mcs : list[list[str]]
        Minimal cut sets.
    """

    top_event: str
    gates: list[dict[str, Any]]
    basic_events: list[dict[str, Any]]
    mcs: list[list[str]]


def generate_fault_tree(
    module_name: str,
    equations: dict[str, str],
) -> FaultTree:
    """Generate FTA/FMEA for DO-254 Level A certification."""
    top = f"{module_name}_SYSTEM_FAILURE"
    basic_events = []
    for sv in equations:
        basic_events.extend(
            [
                {
                    "id": f"{sv}_stuck_at_0",
                    "rate": 1e-7,
                    "description": f"{sv} register stuck-at-0",
                },
                {"id": f"{sv}_overflow", "rate": 1e-6, "description": f"{sv} arithmetic overflow"},
            ]
        )
    basic_events.extend(
        [
            {"id": "clk_failure", "rate": 1e-9, "description": "Clock failure"},
            {"id": "power_glitch", "rate": 1e-8, "description": "Power glitch"},
        ]
    )

    gates = [
        {
            "id": "G1",
            "type": "OR",
            "description": "System failure",
            "inputs": [e["id"] for e in basic_events],
        },
    ]

    # Minimal cut sets: each basic event alone can cause failure (OR gate)
    mcs: list[list[str]] = [[str(e["id"])] for e in basic_events]

    return FaultTree(
        top_event=top,
        gates=gates,
        basic_events=basic_events,
        mcs=mcs,
    )
