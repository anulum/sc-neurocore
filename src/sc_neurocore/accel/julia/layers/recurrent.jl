# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for layers/recurrent

module RecurrentAccel

using Statistics, LinearAlgebra

mutable struct SCRecurrentLayerState
    n_inputs::Float64
    n_neurons::Float64
    feedback_strength::Float64
    input_strength::Float64
    spectral_radius::Float64
    length::Float64
    seed::Float64
end

function SCRecurrentLayerState()
    SCRecurrentLayerState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function step(s::SCRecurrentLayerState, input_vector, Any])
    currents = dot(s.W_in, input_vector) + dot(s.W_rec, s.state)
    new_rates = clamp(currents, 0.0, 1.0)
    s.state = new_rates
    return s.state
end

function reset(s::SCRecurrentLayerState)
    s.state = zeros(s.n_neurons)
end

end # module RecurrentAccel
