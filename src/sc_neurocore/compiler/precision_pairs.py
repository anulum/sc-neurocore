# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Precision pairs

"""Canonical LP/HP precision pairs for dual-datapath execution."""

from __future__ import annotations

# All valid LP/HP pair presets — any (data_width, fraction) pair is valid,
# but these are the canonical presets from PRECISION_PRESETS
PRECISION_PAIRS: list[tuple[tuple[int, int], tuple[int, int]]] = [
    # (LP, HP) — LP must have strictly fewer bits than HP
    ((8, 7), (16, 8)),  # Q1.7 → Q8.8
    ((8, 4), (16, 8)),  # Q4.4 → Q8.8
    ((8, 7), (16, 12)),  # Q1.7 → Q4.12
    ((8, 4), (16, 12)),  # Q4.4 → Q4.12
    ((16, 8), (32, 16)),  # Q8.8 → Q16.16 (default)
    ((16, 8), (32, 12)),  # Q8.8 → Q20.12
    ((16, 8), (32, 24)),  # Q8.8 → Q8.24
    ((16, 12), (32, 16)),  # Q4.12 → Q16.16
    ((16, 12), (32, 24)),  # Q4.12 → Q8.24
    ((16, 15), (32, 16)),  # Q1.15 → Q16.16
    ((18, 9), (32, 16)),  # Q9.9 → Q16.16
    ((18, 9), (36, 18)),  # Q9.9 → Q18.18
    ((24, 12), (32, 16)),  # Q12.12 → Q16.16
    ((24, 12), (36, 18)),  # Q12.12 → Q18.18
    ((27, 13), (36, 18)),  # Q14.13 → Q18.18
]
