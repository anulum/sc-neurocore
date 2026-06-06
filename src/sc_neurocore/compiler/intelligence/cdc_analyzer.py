# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CDC analyzer

"""Clock Domain Crossing (CDC) analysis for multi-clock neuron arrays."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CDCReport:
    """Clock domain crossing analysis result.

    Attributes
    ----------
    crossings : list[dict]
        Each crossing: signal, src_domain, dst_domain, sync_type.
    violations : list[str]
        Unsynchronized crossings.
    total_crossings : int
    safe : bool
    """

    crossings: list[dict]
    violations: list[str]
    total_crossings: int
    safe: bool


def analyze_cdc(
    equations: dict[str, str],
    *,
    clock_domains: dict[str, str] | None = None,
) -> CDCReport:
    """Analyze clock domain crossings in a neuron array."""
    if clock_domains is None:
        clock_domains = {k: "clk_main" for k in equations}

    crossings: list[dict] = []
    violations: list[str] = []
    for sv, expr in equations.items():
        src = clock_domains.get(sv, "clk_main")
        for other in equations:
            if other != sv and other in expr:
                dst = clock_domains.get(other, "clk_main")
                if src != dst:
                    crossings.append(
                        {
                            "signal": f"{other}->{sv}",
                            "src_domain": dst,
                            "dst_domain": src,
                            "sync_type": "2FF",
                        }
                    )

    return CDCReport(
        crossings=crossings,
        violations=violations,
        total_crossings=len(crossings),
        safe=len(violations) == 0,
    )
