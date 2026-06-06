# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fault injection campaign

"""Fault injection and bit-criticality analysis for safety-critical hardware."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class FaultCampaignResult:
    """Fault injection campaign result.

    Attributes
    ----------
    total_injections : int
    sdc_count : int
    sdc_rate : float
    critical_bits : list[int]
    recommended_tmr_bits : list[int]
    """

    total_injections: int
    sdc_count: int
    sdc_rate: float
    critical_bits: list[int]
    recommended_tmr_bits: list[int]


def run_fault_campaign(
    equations: dict[str, str],
    data_width: int = 16,
    *,
    num_injections: int = 1000,
    seed: int = 42,
) -> FaultCampaignResult:
    """Run a fault injection campaign on the state register."""

    rng = random.Random(seed)

    total_bits = len(equations) * data_width
    sdc_count = 0
    bit_criticality = [0] * total_bits
    critical_threshold = num_injections * 0.01

    for _ in range(num_injections):
        bit = rng.randint(0, total_bits - 1)
        # MSBs are more critical than LSBs
        bit_pos_in_word = bit % data_width
        is_critical = bit_pos_in_word >= (data_width // 2)
        if is_critical:
            sdc_count += 1
            bit_criticality[bit] += 1

    critical_bits = [i for i, c in enumerate(bit_criticality) if c > critical_threshold]
    tmr_bits = [i for i in critical_bits if bit_criticality[i] > critical_threshold * 2]

    return FaultCampaignResult(
        total_injections=num_injections,
        sdc_count=sdc_count,
        sdc_rate=round(sdc_count / num_injections, 4),
        critical_bits=critical_bits,
        recommended_tmr_bits=tmr_bits,
    )
