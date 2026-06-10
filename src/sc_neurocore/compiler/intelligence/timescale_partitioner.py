# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multi-timescale partitioner

"""Multi-timescale partitioning utilities for ODE systems.

Identifies fast vs slow dynamics and assigns them to different clock
domains, inserting CDC synchronisers at domain boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimescalePartition:
    """Partitioned ODE system by timescale.

    Attributes
    ----------
    fast_equations : dict[str, str]
        Fast dynamics (membrane, spikes).
    slow_equations : dict[str, str]
        Slow dynamics (adaptation, homeostasis).
    fast_clock_div : int
        Clock divider for fast domain (1 = full speed).
    slow_clock_div : int
        Clock divider for slow domain.
    cdc_signals : list[str]
        Signals requiring clock domain crossing.
    """

    fast_equations: dict[str, str]
    slow_equations: dict[str, str]
    fast_clock_div: int
    slow_clock_div: int
    cdc_signals: list[str]


def partition_timescales(
    equations: dict[str, str],
    time_constants: dict[str, float] | None = None,
    *,
    threshold_ratio: float = 10.0,
) -> TimescalePartition:
    """Partition ODE equations by timescale for multi-clock execution.

    Identifies fast vs slow dynamics and assigns them to different
    clock domains, inserting CDC synchronisers at domain boundaries.

    Parameters
    ----------
    equations : dict[str, str]
        ODE equations.
    time_constants : dict[str, float], optional
        Known time constants per variable (ms). If None, estimated
        from equation structure.
    threshold_ratio : float
        Ratio above which a variable is considered "slow".

    Returns
    -------
    TimescalePartition
        Partitioned system with clock assignments.
    """
    if time_constants is None:
        time_constants = {}
        for sv, expr in equations.items():
            # Heuristic: count operations as proxy for timescale
            ops = expr.count("*") + expr.count("/")
            if ops == 0:
                time_constants[sv] = 1.0  # fast (direct)
            else:
                time_constants[sv] = float(ops)  # slower with more ops

    if not time_constants:
        return TimescalePartition(
            fast_equations=dict(equations),
            slow_equations={},
            fast_clock_div=1,
            slow_clock_div=1,
            cdc_signals=[],
        )

    min_tc = min(time_constants.values())
    fast_eqs = {}
    slow_eqs = {}
    for sv, expr in equations.items():
        tc = time_constants.get(sv, min_tc)
        if tc / min_tc >= threshold_ratio:
            slow_eqs[sv] = expr
        else:
            fast_eqs[sv] = expr

    # If nothing is slow, everything is fast
    if not slow_eqs:
        return TimescalePartition(
            fast_equations=fast_eqs,
            slow_equations={},
            fast_clock_div=1,
            slow_clock_div=1,
            cdc_signals=[],
        )

    # Compute clock divider from ratio
    max_tc = max(time_constants[sv] for sv in slow_eqs)
    slow_div = max(2, int(max_tc / min_tc))

    # Find CDC signals: fast vars referenced in slow equations
    cdc = []
    for sv_fast in fast_eqs:
        for _sv_slow, expr in slow_eqs.items():
            if sv_fast in expr and sv_fast not in cdc:
                cdc.append(sv_fast)

    return TimescalePartition(
        fast_equations=fast_eqs,
        slow_equations=slow_eqs,
        fast_clock_div=1,
        slow_clock_div=slow_div,
        cdc_signals=cdc,
    )
