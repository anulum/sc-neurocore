# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — constant-memory online learning for SNNs

"""Constant-memory online training for spiking neural networks.

Train SNNs on temporal sequences without backpropagation through time.
BPTT stores O(T) activations for a length-T sequence; these methods keep only
per-synapse eligibility traces, so working memory does not grow with sequence
length.

Implemented methods
-------------------
- :class:`EpropTrainer` — e-prop (Bellec et al. 2020), the three-factor
  eligibility-propagation rule for a single-layer recurrent SNN.
- :class:`OnlineTrainer` / :class:`OnlineLIFLayer` — an eligibility-based local
  update for a feedforward LIF stack driven by top-down learning signals.
"""

from .eprop import EpropTrainer
from .online_trainer import OnlineLIFLayer, OnlineTrainer

__all__ = ["EpropTrainer", "OnlineLIFLayer", "OnlineTrainer"]
