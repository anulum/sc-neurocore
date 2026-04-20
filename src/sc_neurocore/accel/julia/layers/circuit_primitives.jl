# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for layers/circuit_primitives

module CircuitPrimitivesAccel

using Statistics, LinearAlgebra

mutable struct WinnerTakeAllState
    n_neurons::Float64
    inhibition_strength::Float64
    radius::Float64
    k::Float64
end

function WinnerTakeAllState()
    WinnerTakeAllState(0.0, 0.3, 2.0, 1.0)
end

function apply(s::WinnerTakeAllState, rates)
    inhibition = s._kernel @ rates
    return max(rates - inhibition, 0.0)
end

function apply(s::WinnerTakeAllState, rates)
    if s.k >= s.n_neurons
        return rates.copy()
    top_k = np.argsort(rates)[-s.k :]
    result = np.zeros_like(rates)
    result[top_k] = rates[top_k]
    return result
end

function winners(s::WinnerTakeAllState, rates)
    return np.argsort(rates)[-s.k :][::-1]
end

end # module CircuitPrimitivesAccel
