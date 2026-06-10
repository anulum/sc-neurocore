# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Side-channel leakage lint

"""Analyse equations for power/timing side-channel vulnerabilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SideChannelFinding:
    """Side-channel leakage finding.

    Attributes
    ----------
    signal : str
        Signal name.
    risk_level : str
        ``"high"``, ``"medium"``, or ``"low"``.
    category : str
        ``"timing"`` or ``"power"``.
    description : str
        Explanation.
    recommendation : str
        Mitigation suggestion.
    """

    signal: str
    risk_level: str
    category: str
    description: str
    recommendation: str


def lint_side_channels(
    equations: dict[str, str],
    *,
    module_name: str = "neuron",
    data_width: int = 16,
) -> list[SideChannelFinding]:
    """Analyse equations for power/timing side-channel vulnerabilities."""
    findings = []

    for sv, expr in equations.items():
        if "if" in expr or "?" in expr:
            findings.append(
                SideChannelFinding(
                    signal=sv,
                    risk_level="high",
                    category="timing",
                    description=f"Data-dependent branch in {sv} equation",
                    recommendation="Use constant-time mux instead of branch",
                )
            )

        if "/" in expr:
            findings.append(
                SideChannelFinding(
                    signal=sv,
                    risk_level="medium",
                    category="timing",
                    description=f"Division in {sv} — variable latency",
                    recommendation="Use fixed-point shift or LUT-based reciprocal",
                )
            )

        if "*" in expr:
            findings.append(
                SideChannelFinding(
                    signal=sv,
                    risk_level="low",
                    category="power",
                    description=f"Multiply in {sv} — Hamming weight leakage",
                    recommendation="Add random masking for security-critical paths",
                )
            )

    findings.append(
        SideChannelFinding(
            signal="spike_out",
            risk_level="medium",
            category="power",
            description="Spike output toggles are data-dependent",
            recommendation="Add constant-activity output buffer",
        )
    )

    return findings
