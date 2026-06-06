# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Regression watchdog

"""Detect performance and resource regressions between compilations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RegressionCheck:
    """Compilation regression check result.

    Attributes
    ----------
    metric : str
    baseline : float
    current : float
    delta_pct : float
    regression : bool
    """

    metric: str
    baseline: float
    current: float
    delta_pct: float
    regression: bool


def check_regression(
    baseline: dict[str, float],
    current: dict[str, float],
    *,
    threshold_pct: float = 5.0,
) -> list[RegressionCheck]:
    """Detect performance regressions between compilations."""
    results = []
    for metric, base_val in baseline.items():
        cur_val = current.get(metric, base_val)
        if base_val != 0:
            delta = ((cur_val - base_val) / abs(base_val)) * 100
        else:
            delta = 0.0
        results.append(
            RegressionCheck(
                metric=metric,
                baseline=base_val,
                current=cur_val,
                delta_pct=round(delta, 2),
                regression=abs(delta) > threshold_pct,
            )
        )
    return results
