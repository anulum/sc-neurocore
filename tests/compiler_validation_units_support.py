# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_compiler_validation_units.py

from __future__ import annotations

"""Branch-level contracts for adaptive-runtime precision coercion and the
low-precision / high-precision datapath and hysteresis validators."""
import math
import pytest
from sc_neurocore.compiler.quantizer import BlockFloatingMode, QFormat
from sc_neurocore.compiler.validation import (
    _coerce_precision,
    _validate_hysteresis,
    _validate_lp_hp,
)

__all__ = [
    "math",
    "pytest",
    "BlockFloatingMode",
    "QFormat",
    "_coerce_precision",
    "_validate_hysteresis",
    "_validate_lp_hp",
]
