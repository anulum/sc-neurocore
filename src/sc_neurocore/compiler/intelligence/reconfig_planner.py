# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Partial reconfiguration planner

"""FPGA partial reconfiguration planning utilities.

Plans the splitting of neuron arrays across dynamic partial
reconfiguration (DPR) partitions for time-multiplexed execution.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReconfigPartition:
    """Partial reconfiguration partition plan.

    Attributes
    ----------
    partitions : list[dict[str, list[str]]]
        Each partition maps region name → assigned variables.
    schedule : list[str]
        Time-ordered bitstream swap schedule.
    total_regions : int
        Number of reconfigurable regions.
    bitstream_count : int
        Total partial bitstreams needed.
    """

    partitions: list[dict[str, list[str]]]
    schedule: list[str]
    total_regions: int
    bitstream_count: int


def plan_partial_reconfiguration(
    equations: dict[str, str],
    *,
    max_regions: int = 4,
    time_slots: int = 2,
) -> ReconfigPartition:
    """Plan FPGA partial reconfiguration for SNN time-multiplexing.

    Splits neuron equations across reconfigurable regions and
    generates a swap schedule.

    Parameters
    ----------
    equations : dict[str, str]
        ODE equations.
    max_regions : int
        Maximum reconfigurable regions.
    time_slots : int
        Number of time-multiplexed slots.

    Returns
    -------
    ReconfigPartition
        Partition plan with schedule.
    """
    vars_list = list(equations.keys())
    regions = min(max_regions, len(vars_list))

    # Distribute variables across regions
    partitions = []
    for slot in range(time_slots):
        partition: dict[str, list[str]] = {}
        for i, sv in enumerate(vars_list):
            region = f"region_{i % regions}"
            if region not in partition:
                partition[region] = []
            partition[region].append(sv)
        partitions.append(partition)

    # Generate swap schedule
    schedule = []
    for slot in range(time_slots):
        schedule.append(f"slot_{slot}: load bitstream_{slot}, activate {regions} region(s)")

    return ReconfigPartition(
        partitions=partitions,
        schedule=schedule,
        total_regions=regions,
        bitstream_count=time_slots,
    )
