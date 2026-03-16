# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""sc_neurocore.drivers -- Tier: research (experimental / research)."""

__tier__ = "research"

from .sc_neurocore_driver import SC_NeuroCore_Driver
from .physical_twin import PhysicalTwinBridge
from .verify_hardware_link import verify_link

__all__ = [
    "SC_NeuroCore_Driver",
    "PhysicalTwinBridge",
    "verify_link",
]
