# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for neurons/dendritic

module DendriticAccel

using Statistics, LinearAlgebra

mutable struct StochasticDendriticNeuronState
    threshold::Float64
    _last_current::Float64
end

function StochasticDendriticNeuronState()
    StochasticDendriticNeuronState(0.0, 0.0)
end

function step(s::StochasticDendriticNeuronState, input_a, input_b)
    d1 = input_a
    d2 = input_b
    # XOR nonlinearity: d1 + d2 - 2*d1*d2
    current = d1 + d2 - 2.0 * (d1 * d2)
    s._last_current = current
    if current > s.threshold
        return 1
    return 0
end

function reset_state(s::StochasticDendriticNeuronState)
    s._last_current = 0.0
end

function get_state(s::StochasticDendriticNeuronState)
    return Dict("last_current": s._last_current, "threshold": s.threshold)
end

end # module DendriticAccel
end
