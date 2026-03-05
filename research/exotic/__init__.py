# SPDX-License-Identifier: AGPL-3.0-or-later
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
