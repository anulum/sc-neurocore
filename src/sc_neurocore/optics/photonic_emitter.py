# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Photonic compatibility facade

"""Stable photonic compiler, co-simulation, layout, and crosstalk API.

Implementation ownership is split by responsibility across private sibling
modules. Historical imports, class identities, and pickle-qualified paths
remain anchored here.
"""

from __future__ import annotations

from ._photonic_compiler import CompilationResult, PhotonicCompiler
from ._photonic_conversion import BitstreamToOptical
from ._photonic_crosstalk import (
    CrosstalkModel,
    WaveguidePair,
    _HAS_RUST_PH as _HAS_RUST_PH,
    py_ph_analyze_crosstalk as py_ph_analyze_crosstalk,
    py_ph_analyze_crosstalk_bank as py_ph_analyze_crosstalk_bank,
    py_ph_analyze_crosstalk_pairs as py_ph_analyze_crosstalk_pairs,
)
from ._photonic_emitter import PhotonicEmitter
from ._photonic_fdtd import FDTD2DSolver, FDTDSolver
from ._photonic_meep import MeepAdapter
from ._photonic_types import OpticalModulation, OpticalPulse, PhotonicTarget

__all__ = [
    "BitstreamToOptical",
    "CompilationResult",
    "CrosstalkModel",
    "FDTD2DSolver",
    "FDTDSolver",
    "MeepAdapter",
    "OpticalModulation",
    "OpticalPulse",
    "PhotonicCompiler",
    "PhotonicEmitter",
    "PhotonicTarget",
    "WaveguidePair",
]

for _public_class in (
    BitstreamToOptical,
    CompilationResult,
    CrosstalkModel,
    FDTD2DSolver,
    FDTDSolver,
    MeepAdapter,
    OpticalModulation,
    OpticalPulse,
    PhotonicCompiler,
    PhotonicEmitter,
    PhotonicTarget,
    WaveguidePair,
):
    _public_class.__module__ = __name__

del _public_class
