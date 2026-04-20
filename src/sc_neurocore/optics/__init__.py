# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.optics

"""sc_neurocore.optics -- Photonic computation, compilation and physical modeling.

Requires `pip install "sc-neurocore[optics]"` for gdsfactory + full Rust paths.
"""

from .photonic_layer import PhotonicBitstreamLayer
from .photonic_emitter import (
    PhotonicEmitter,
    PhotonicTarget,
    PhotonicCompiler,
    CompilationResult,
    OpticalModulation,
    BitstreamToOptical,
    FDTDSolver,
    FDTD2DSolver,
    CrosstalkModel,
    WaveguidePair,
    OpticalPulse,
    MeepAdapter,
)

__all__ = [
    "PhotonicBitstreamLayer",
    "PhotonicEmitter",
    "PhotonicTarget",
    "PhotonicCompiler",
    "CompilationResult",
    "OpticalModulation",
    "BitstreamToOptical",
    "FDTDSolver",
    "FDTD2DSolver",
    "CrosstalkModel",
    "WaveguidePair",
    "OpticalPulse",
    "MeepAdapter",
]
