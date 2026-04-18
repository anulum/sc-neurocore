# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Identity continuity substrate

"""Identity continuity substrate: persistent spiking network, encoding, decoding, checkpointing."""

from .substrate import IdentitySubstrate
from .encoder import TraceEncoder
from .decoder import StateDecoder
from .checkpoint import Checkpoint
from .director import DirectorController

__all__ = [
    "IdentitySubstrate",
    "TraceEncoder",
    "StateDecoder",
    "Checkpoint",
    "DirectorController",
]
