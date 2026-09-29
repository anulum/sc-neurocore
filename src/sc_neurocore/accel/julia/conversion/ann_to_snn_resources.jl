# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia dense IF numeric resource admission

"""Require a positive addressable numeric working limit; excludes caller/runtime storage."""
function working_limit(value::Int)
    value > 0 || throw(ArgumentError("working byte budget must be positive"))
    return value
end

"""Admit eight bytes per 2P + 2F + 2S + 2O + H + 5M + I before allocation."""
function admit_replay(layers, steps::Int, batch::Int, trace::Bool, linear::Bool, maximum::Int)
    working_limit(maximum)
    coefficients = sum(BigInt(length(l.weights)) + (isnothing(l.bias) ? 0 : length(l.bias)) for l in layers)
    nodes = sum(BigInt(l.outputs) for l in layers)
    states = BigInt(batch) * nodes
    output = BigInt(batch) * layers[end].outputs
    largest = BigInt(batch) * maximum_width(layers)
    frame = BigInt(batch) * layers[1].inputs
    frames = BigInt(steps) * frame
    spiking = linear ? nodes - layers[end].outputs : nodes
    traces = trace ? BigInt(steps) * batch * (nodes + spiking) : BigInt(0)
    bytes = 8 * (2coefficients + 2frames + 2states + 2output + traces + 5largest + frame)
    bytes <= maximum || throw(OutOfMemoryError())
    return nothing
end

"""Return the largest output width in the checked nonempty dense stack."""
maximum_width(layers) = maximum(l.outputs for l in layers)
