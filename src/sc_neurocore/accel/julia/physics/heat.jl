# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia mirror for physics/heat contracts

module HeatAccel

using Random
using Statistics

mutable struct FeynmanKacHeatSolverState
    length::Float64
    diffusivity::Float64
    dt::Float64
    walkers::Vector{Float64}
end

function FeynmanKacHeatSolverState(length::Real, diffusivity::Real, dt::Real, walkers::Vector{Float64})
    if !isfinite(float(length)) || float(length) <= 0.0
        throw(ArgumentError("length must be a finite positive number"))
    end
    if !isfinite(float(diffusivity)) || float(diffusivity) < 0.0
        throw(ArgumentError("diffusivity must be a finite non-negative number"))
    end
    if !isfinite(float(dt)) || float(dt) <= 0.0
        throw(ArgumentError("dt must be a finite positive number"))
    end
    if any(!isfinite, walkers) || any(x -> x < 0.0 || x > float(length), walkers)
        throw(ArgumentError("walkers must be finite values inside [0, length]"))
    end
    FeynmanKacHeatSolverState(float(length), float(diffusivity), float(dt), copy(walkers))
end

function reflect_into_interval(x::Vector{Float64}, length::Float64)::Vector{Float64}
    period = 2.0 * length
    folded = mod.(x, period)
    return ifelse.(folded .<= length, folded, period .- folded)
end

function step!(s::FeynmanKacHeatSolverState, rng::AbstractRNG=Random.default_rng())
    sigma = sqrt(2.0 * s.diffusivity * s.dt)
    s.walkers .+= randn(rng, length(s.walkers)) .* sigma
    s.walkers = reflect_into_interval(s.walkers, s.length)
    return s
end

const StochasticHeatSolverState = FeynmanKacHeatSolverState

end # module HeatAccel
