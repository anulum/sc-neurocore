# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.interfaces -- Tier: research (experimental

"""sc_neurocore.interfaces -- Tier: research (experimental / research)."""

__tier__ = "research"

from .bci import BCIDecoder, BCIEncoder
from .bci_closed_loop import (
    ClosedLoopBCIConfig,
    ClosedLoopBCIResult,
    ClosedLoopBCITemplate,
    FeedbackFrame,
    ImplantEmulator,
    RateSpikeDecoder,
)
from .bci_hil_manifest import (
    BCIHILBoardProfile,
    available_bci_hil_profiles,
    build_bci_hil_reference_manifest,
    create_bci_hil_template,
    get_bci_hil_profile,
)
from .dvs_input import DVSInputLayer
from .real_world import LSLBridge, ROS2Node
from .zenith_bci_loop import ZenithBCILoop, ZenithBCILoopConfig, ZenithBCILoopResult

__all__ = [
    "BCIEncoder",
    "BCIDecoder",
    "ClosedLoopBCIConfig",
    "ClosedLoopBCIResult",
    "ClosedLoopBCITemplate",
    "FeedbackFrame",
    "ImplantEmulator",
    "RateSpikeDecoder",
    "BCIHILBoardProfile",
    "available_bci_hil_profiles",
    "build_bci_hil_reference_manifest",
    "create_bci_hil_template",
    "get_bci_hil_profile",
    "DVSInputLayer",
    "LSLBridge",
    "ROS2Node",
    "ZenithBCILoop",
    "ZenithBCILoopConfig",
    "ZenithBCILoopResult",
]
