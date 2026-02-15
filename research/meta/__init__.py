"""sc_neurocore.meta -- Tier: contrib (speculative / theoretical)."""

__tier__ = "contrib"

from .black_hole import EventHorizonLayer
from .dao import AgentDAO, Proposal
from .fermi_game import DarkForestAgent
from .hyper_turing import OracleLayer
from .omega import OmegaIntegrator
from .singularity import RecursiveSelfImprover
from .time_crystal import TimeCrystalLayer
from .time_travel import CTCLayer
from .vacuum import VacuumNoiseSource

__all__ = [
    "EventHorizonLayer",
    "AgentDAO",
    "Proposal",
    "DarkForestAgent",
    "OracleLayer",
    "OmegaIntegrator",
    "RecursiveSelfImprover",
    "TimeCrystalLayer",
    "CTCLayer",
    "VacuumNoiseSource",
]
