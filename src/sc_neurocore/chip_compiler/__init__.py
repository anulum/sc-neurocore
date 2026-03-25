# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multi-chip neuromorphic hardware compiler

"""Target-agnostic SNN compiler: map networks to neuromorphic chips via YAML specs."""

from .chip_spec import ChipSpec, CoreSpec, load_chip_spec, BUILTIN_CHIPS
from .compiler import compile_for_chip, CompilationResult, CoreMapping

__all__ = [
    "ChipSpec",
    "CoreSpec",
    "load_chip_spec",
    "BUILTIN_CHIPS",
    "compile_for_chip",
    "CompilationResult",
    "CoreMapping",
]
