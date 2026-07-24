# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_chip_compiler.py

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.chip_compiler import (
    ChipSpec,
    CoreSpec,
    BUILTIN_CHIPS,
    compile_for_chip,
    CompilationResult,
)

__all__ = [
    "np",
    "pytest",
    "ChipSpec",
    "CoreSpec",
    "BUILTIN_CHIPS",
    "compile_for_chip",
    "CompilationResult",
]
