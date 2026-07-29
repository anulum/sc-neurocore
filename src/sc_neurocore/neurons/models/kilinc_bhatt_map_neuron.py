# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — deprecated compatibility identity

"""Compatibility import for the former unsupported model name.

``KilincBhattMapNeuron`` is an alias, not a distinct scientific model. New
code must use :class:`SCAdaptiveThresholdMapNeuron`.
"""

from sc_neurocore.neurons.models.sc_adaptive_threshold_map_neuron import (
    SCAdaptiveThresholdMapNeuron,
    SCAdaptiveThresholdMapResult,
)

KilincBhattMapNeuron = SCAdaptiveThresholdMapNeuron
KilincBhattMapResult = SCAdaptiveThresholdMapResult

__all__ = ["KilincBhattMapNeuron", "KilincBhattMapResult"]
