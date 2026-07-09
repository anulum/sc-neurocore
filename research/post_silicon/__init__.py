# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""sc_neurocore.post_silicon -- Tier: contrib (speculative / theoretical)."""

__tier__ = "contrib"

from .claytronics import CatomLattice
from .femto import FemtoSwitch
from .reversible import ReversibleLayer
from .synthetic_cell import CellularComputer

__all__ = [
    "CatomLattice",
    "FemtoSwitch",
    "ReversibleLayer",
    "CellularComputer",
]
