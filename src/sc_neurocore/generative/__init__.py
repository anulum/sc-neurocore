# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.generative -- Tier: research (experimental

"""sc_neurocore.generative -- Tier: research (experimental / research)."""

__tier__ = "research"

from .audio_synthesis import SCAudioSynthesizer
from .text_gen import SCTextGenerator
from .three_d_gen import SC3DGenerator

__all__ = [
    "SCAudioSynthesizer",
    "SCTextGenerator",
    "SC3DGenerator",
]
