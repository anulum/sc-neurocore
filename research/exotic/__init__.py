# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""sc_neurocore.exotic -- Tier: contrib (speculative / theoretical)."""

__tier__ = "contrib"

from .anyon import AnyonBraidLayer
from .chemical import ReactionDiffusionSolver
from .constructor import ConstructorCell
from .dyson_grid import DysonPowerGrid
from .fungal import MyceliumLayer
from .matrioshka import DysonSwarmNet
from .mechanical import MechanicalLatticeLayer
from .space import RadHardLayer

__all__ = [
    "AnyonBraidLayer",
    "ReactionDiffusionSolver",
    "ConstructorCell",
    "DysonPowerGrid",
    "MyceliumLayer",
    "DysonSwarmNet",
    "MechanicalLatticeLayer",
    "RadHardLayer",
]
