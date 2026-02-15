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
