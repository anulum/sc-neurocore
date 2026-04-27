# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.hdl_gen -- Tier: research (experimental /

"""sc_neurocore.hdl_gen -- Tier: research (experimental / research)."""

__tier__ = "research"

from .verilog_generator import VerilogGenerator
from .spice_generator import SpiceGenerator
from .aer_emitter import AEREmitter
from .kuramoto_emitter import KuramotoEmitter
from .lfsr16_emitter import Lfsr16Emitter
from .sobol16_emitter import Sobol16Emitter

__all__ = [
    "VerilogGenerator",
    "SpiceGenerator",
    "AEREmitter",
    "KuramotoEmitter",
    "Lfsr16Emitter",
    "Sobol16Emitter",
]
