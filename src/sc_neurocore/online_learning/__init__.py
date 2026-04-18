# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — O(1) memory online learning for SNNs

"""O(1) memory training: e-prop, RTRL, and forward-gradient methods.

Train SNNs on long temporal sequences without unrolling through time.
BPTT requires O(T) memory; these methods require O(1) per timestep.
"""

from .eprop import EpropTrainer
from .online_trainer import OnlineLIFLayer, OnlineTrainer

__all__ = ["EpropTrainer", "OnlineLIFLayer", "OnlineTrainer"]
