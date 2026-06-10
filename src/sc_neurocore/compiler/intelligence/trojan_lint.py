# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hardware trojan lint

"""Detect suspicious combinational paths that could hide hardware trojans."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrojanLintResult:
    """Hardware trojan lint analysis result.

    Attributes
    ----------
    suspicious_paths : list[str]
    risk_level : str
    total_checks : int
    """

    suspicious_paths: list[str]
    risk_level: str
    total_checks: int


def lint_hardware_trojans(
    equations: dict[str, str],
    *,
    check_dormant: bool = True,
    check_payload: bool = True,
) -> TrojanLintResult:
    """Detect suspicious combinational paths that could hide trojans."""
    suspicious = []
    checks = 0

    for var, expr in equations.items():
        checks += 1
        if check_dormant and ("if" in expr or "?" in expr):
            suspicious.append(f"{var}: conditional path detected — potential dormant trigger")
        if check_payload:
            other_vars = [v for v in equations if v != var]
            for ov in other_vars:
                if ov in expr:
                    checks += 1

    risk = "LOW"
    if len(suspicious) >= 2:
        risk = "HIGH"
    elif len(suspicious) >= 1:
        risk = "MEDIUM"

    return TrojanLintResult(
        suspicious_paths=suspicious,
        risk_level=risk,
        total_checks=checks,
    )
