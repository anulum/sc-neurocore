# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.hdl_gen -- Tier: research (experimental /
"""sc_neurocore.hdl_gen -- Tier: research (experimental / research)."""

__tier__ = "research"

from .verilog_generator import VerilogGenerator, emit_sources_from_ir
from .spice_generator import SpiceGenerator
from .aer_emitter import AEREmitter
from .kuramoto_emitter import KuramotoEmitter
from .lfsr16_emitter import Lfsr16Emitter
from .side_channel_encoding_emitter import (
    SIDE_CHANNEL_HDL_HOOK_SCHEMA_VERSION,
    SideChannelEncodingEmitter,
)
from .online_learning_emitter import (
    ONLINE_O1_HDL_MANIFEST_SCHEMA_VERSION,
    ONLINE_O1_RESOURCE_ESTIMATE_SCHEMA_VERSION,
    OnlineO1LearningEmitter,
    OnlineO1ResourceEstimate,
)
from .sobol16_emitter import Sobol16Emitter
from .quasirandom_emitter import QuasiRandomEmitter, Halton16Emitter
from .tmr_wrapper import generate_tmr_wrapper
from .bus_interface import generate_live_parameter_bank
from .ip_xact import generate_ip_xact

__all__ = [
    "VerilogGenerator",
    "emit_sources_from_ir",
    "SpiceGenerator",
    "AEREmitter",
    "KuramotoEmitter",
    "Lfsr16Emitter",
    "SIDE_CHANNEL_HDL_HOOK_SCHEMA_VERSION",
    "SideChannelEncodingEmitter",
    "ONLINE_O1_HDL_MANIFEST_SCHEMA_VERSION",
    "ONLINE_O1_RESOURCE_ESTIMATE_SCHEMA_VERSION",
    "OnlineO1LearningEmitter",
    "OnlineO1ResourceEstimate",
    "Sobol16Emitter",
    "QuasiRandomEmitter",
    "Halton16Emitter",
    "generate_tmr_wrapper",
    "generate_live_parameter_bank",
    "generate_ip_xact",
]
