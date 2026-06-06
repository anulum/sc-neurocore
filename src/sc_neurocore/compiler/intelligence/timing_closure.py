# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Formal timing closure

"""Static timing analysis and timing closure verification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TimingReport:
    """Static timing analysis report.

    Attributes
    ----------
    critical_path : list[str]
    critical_delay_ns : float
    target_period_ns : float
    slack_ns : float
    timing_met : bool
    recommendations : list[str]
    """

    critical_path: list[str]
    critical_delay_ns: float
    target_period_ns: float
    slack_ns: float
    timing_met: bool
    recommendations: list[str]


def verify_timing_closure(
    equations: dict[str, str],
    *,
    target_freq_mhz: float = 250.0,
    data_width: int = 16,
) -> TimingReport:
    """Perform static timing analysis on the dataflow graph."""
    target_period = 1000.0 / target_freq_mhz

    # Model operator delays (ns)
    add_delay = 0.3 * (data_width / 16)
    mul_delay = 1.2 * (data_width / 16)

    # Estimate critical path
    path = []
    total_delay = 0.0
    for var, expr in equations.items():
        ops = expr.count("+") + expr.count("-")
        muls = expr.count("*")
        var_delay = ops * add_delay + muls * mul_delay + 0.5
        path.append(f"{var}({ops}add+{muls}mul)")
        total_delay = max(total_delay, var_delay)

    slack = target_period - total_delay
    recs = []
    if slack < 0:
        stages_needed = int(-slack / target_period) + 2
        recs.append(f"Insert {stages_needed} pipeline stages")
        recs.append(f"Or reduce frequency to {int(1000 / total_delay)} MHz")
    elif slack < target_period * 0.1:
        recs.append("Tight slack — consider adding 1 pipeline stage")

    return TimingReport(
        critical_path=path,
        critical_delay_ns=round(total_delay, 3),
        target_period_ns=round(target_period, 3),
        slack_ns=round(slack, 3),
        timing_met=slack >= 0,
        recommendations=recs,
    )
