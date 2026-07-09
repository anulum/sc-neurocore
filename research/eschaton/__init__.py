# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""sc_neurocore.eschaton -- Tier: contrib (speculative / theoretical)."""

__tier__ = "contrib"

from .computronium import PlanckGrid
from .heat_death import HeatDeathLayer
from .holographic import HolographicBoundary
from .simulation import NestedUniverse

__all__ = [
    "PlanckGrid",
    "HeatDeathLayer",
    "HolographicBoundary",
    "NestedUniverse",
]
