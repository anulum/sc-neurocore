# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multi-objective Pareto explorer

"""Explore power/area/latency tradeoffs in design space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ParetoPoint:
    """A single Pareto-optimal design point.

    Attributes
    ----------
    config : dict
    power_mw : float
    area_luts : int
    latency_ns : float
    """

    config: dict[str, Any]
    power_mw: float
    area_luts: int
    latency_ns: float


def explore_pareto(
    equations: dict[str, str],
    *,
    widths: list[int] | None = None,
    pipeline_depths: list[int] | None = None,
) -> list[ParetoPoint]:
    """Explore power/area/latency Pareto frontier."""
    if widths is None:
        widths = [8, 16, 24, 32]
    if pipeline_depths is None:
        pipeline_depths = [1, 2, 4]

    n_vars = len(equations)
    points = []
    for w in widths:
        for d in pipeline_depths:
            power = n_vars * (w / 8) ** 1.5 * (1.0 / d) * 10
            area = n_vars * w * d * 3
            latency = 1000.0 / (d * (32 / w))
            points.append(
                ParetoPoint(
                    config={"data_width": w, "pipeline_depth": d},
                    power_mw=round(power, 2),
                    area_luts=area,
                    latency_ns=round(latency, 2),
                )
            )

    # Filter non-dominated
    pareto = []
    for p in points:
        dominated = False
        for q in points:
            if (
                q.power_mw <= p.power_mw
                and q.area_luts <= p.area_luts
                and q.latency_ns <= p.latency_ns
                and (
                    q.power_mw < p.power_mw
                    or q.area_luts < p.area_luts
                    or q.latency_ns < p.latency_ns
                )
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(p)

    return sorted(pareto, key=lambda p: p.power_mw)
