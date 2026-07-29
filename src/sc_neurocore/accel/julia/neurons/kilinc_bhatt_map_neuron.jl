# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — deprecated Julia compatibility identity

"""Compatibility namespace for the former unsupported model name."""
module KilincBhattMapNeuronAccel

include("sc_adaptive_threshold_map_neuron.jl")
using .SCAdaptiveThresholdMapNeuronAccel

const KilincBhattMapNeuronState = SCAdaptiveThresholdMapNeuronState

export KilincBhattMapNeuronState, reset!, step!, valid

end # module KilincBhattMapNeuronAccel
