# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Synapses Package Init

"""Expose synapse and plasticity primitives for stochastic networks."""

from .sc_synapse import BitstreamSynapse
from .dot_product import BitstreamDotProduct
from .stochastic_stdp import StochasticSTDPSynapse
from .r_stdp import RewardModulatedSTDPSynapse
from .triplet_stdp import TripletSTDP
from .bcm import BCMSynapse
from .clopath_stdp import ClopathSTDP
from .tripartite import TripartiteSynapse
from .short_term_plasticity import ShortTermPlasticitySynapse
from .dopamine_stdp import DopamineStdpSynapse

__all__ = [
    "BitstreamSynapse",
    "BitstreamDotProduct",
    "StochasticSTDPSynapse",
    "RewardModulatedSTDPSynapse",
    "TripletSTDP",
    "BCMSynapse",
    "ClopathSTDP",
    "TripartiteSynapse",
    "ShortTermPlasticitySynapse",
    "DopamineStdpSynapse",
]
