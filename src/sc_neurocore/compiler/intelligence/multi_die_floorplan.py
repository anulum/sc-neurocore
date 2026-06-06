# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multi-die floorplanner

"""Floorplanning and die assignment for multi-die/chiplet SNN systems."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FloorplanResult:
    """Multi-die/chiplet floorplan assignment.

    Attributes
    ----------
    die_assignment : dict[str, int]
        Block name → die index.
    die_utilization : dict[int, float]
        Die index → utilization (0-1).
    total_dies : int
    """

    die_assignment: dict[str, int]
    die_utilization: dict[int, float]
    total_dies: int


def plan_multi_die_floorplan(
    blocks: dict[str, int],
    *,
    die_capacity: int = 1000,
    num_dies: int = 4,
) -> FloorplanResult:
    """Assign neuron blocks to chiplet/die positions.

    Uses first-fit-decreasing bin packing.

    Parameters
    ----------
    blocks : dict[str, int]
        Block name → neuron count.
    die_capacity : int
        Max neurons per die.
    num_dies : int
        Available dies.

    Returns
    -------
    FloorplanResult
    """
    sorted_blocks = sorted(blocks.items(), key=lambda x: x[1], reverse=True)
    assignment: dict[str, int] = {}
    die_used = [0] * num_dies

    for name, count in sorted_blocks:
        placed = False
        for d in range(num_dies):
            if die_used[d] + count <= die_capacity:
                assignment[name] = d
                die_used[d] += count
                placed = True
                break
        if not placed:
            assignment[name] = num_dies - 1
            die_used[num_dies - 1] += count

    util = {d: round(die_used[d] / die_capacity, 3) for d in range(num_dies) if die_used[d] > 0}

    return FloorplanResult(
        die_assignment=assignment,
        die_utilization=util,
        total_dies=len(util),
    )
