# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive runtime precision facade

"""Dual-datapath adaptive-precision compilation facade."""

from __future__ import annotations

from .compiler_impl import (
    compile_adaptive_precision,
)
from .manifest_gen import (
    _precision_label,
    _precision_manifest,
)
from .precision_pairs import (
    PRECISION_PAIRS,
)
from .validation import (
    _coerce_precision,
    _validate_hysteresis,
    _validate_lp_hp,
)

__all__ = [
    "PRECISION_PAIRS",
    "compile_adaptive_precision",
]
