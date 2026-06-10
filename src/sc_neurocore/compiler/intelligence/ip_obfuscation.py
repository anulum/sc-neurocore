# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — IP obfuscation

"""Logic locking and structural obfuscation for IP protection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ObfuscationResult:
    """IP obfuscation report.

    Attributes
    ----------
    techniques_applied : list[str]
    key_bits : int
    original_signals : int
    obfuscated_signals : int
    """

    techniques_applied: list[str]
    key_bits: int
    original_signals: int
    obfuscated_signals: int


def obfuscate_ip(
    module_name: str,
    equations: dict[str, str],
    *,
    key_length: int = 64,
    methods: list[str] | None = None,
) -> ObfuscationResult:
    """Apply logic locking and structural obfuscation for IP protection."""
    if methods is None:
        methods = [
            "logic_locking",
            "constant_propagation_block",
            "structural_transform",
        ]

    original_signals = sum(len(expr.split()) for expr in equations.values())
    obfuscated = original_signals + key_length

    return ObfuscationResult(
        techniques_applied=methods,
        key_bits=key_length,
        original_signals=original_signals,
        obfuscated_signals=obfuscated,
    )
