# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

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
