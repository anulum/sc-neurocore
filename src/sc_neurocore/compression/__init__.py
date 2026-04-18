# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SNN model compression toolkit

"""SNN model compression: pruning, quantization, clustering."""

from .pruning import prune_weights, prune_neurons, prune_stochastic, PruningReport
from .quantization import quantize_delays, quantize_weights

__all__ = [
    "prune_weights",
    "prune_neurons",
    "prune_stochastic",
    "PruningReport",
    "quantize_delays",
    "quantize_weights",
]
