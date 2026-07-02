# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia safety mirror for adapters_holonomic/l6_plan

module L6PlanAccel

using Random
using Statistics

export L6_PlanetaryAdapterState,
    decode,
    encode,
    gaia_kernel,
    get_metrics,
    project_inputs,
    step_jax,
    validate_l6_plan,
    validate_params

mutable struct L6_PlanetaryAdapterState
    n_regions::Int
    bitstream_length::Int
    f_schumann::Float64
    q_factor::Float64
    alpha_gaia::Float64
    p_percolation::Float64
    rng::MersenneTwister
    phi_planetary::Vector{Float64}
    regional_coherence::Vector{Float64}
    t::Float64
end

function L6_PlanetaryAdapterState(;
    n_regions::Int = 100,
    bitstream_length::Int = 1024,
    f_schumann::Real = 7.83,
    q_factor::Real = 4.0,
    alpha_gaia::Real = 0.05,
    p_percolation::Real = 0.592,
    seed::Int = 46,
)::L6_PlanetaryAdapterState
    schumann = Float64(f_schumann)
    quality = Float64(q_factor)
    alpha = Float64(alpha_gaia)
    percolation = Float64(p_percolation)
    validate_params(n_regions, bitstream_length, schumann, quality, alpha, percolation)
    return L6_PlanetaryAdapterState(
        n_regions,
        bitstream_length,
        schumann,
        quality,
        alpha,
        percolation,
        MersenneTwister(seed),
        zeros(Float64, n_regions),
        fill(0.1, n_regions),
        0.0,
    )
end

function validate_params(
    n_regions::Int,
    bitstream_length::Int,
    f_schumann::Real,
    q_factor::Real,
    alpha_gaia::Real,
    p_percolation::Real,
)::Nothing
    n_regions > 0 || throw(ArgumentError("n_regions must be positive."))
    bitstream_length > 0 || throw(ArgumentError("bitstream_length must be positive."))
    isfinite(f_schumann) && f_schumann > 0.0 ||
        throw(ArgumentError("f_schumann must be finite and positive."))
    isfinite(q_factor) && q_factor > 0.0 ||
        throw(ArgumentError("q_factor must be finite and positive."))
    isfinite(alpha_gaia) && alpha_gaia > 0.0 ||
        throw(ArgumentError("alpha_gaia must be finite and positive."))
    isfinite(p_percolation) && 0.0 < p_percolation < 1.0 ||
        throw(ArgumentError("p_percolation must be finite and in (0, 1)."))
    return nothing
end

function validate_l6_plan(state::L6_PlanetaryAdapterState)::Bool
    try
        validate_params(
            state.n_regions,
            state.bitstream_length,
            state.f_schumann,
            state.q_factor,
            state.alpha_gaia,
            state.p_percolation,
        )
    catch
        return false
    end
    return length(state.phi_planetary) == state.n_regions &&
           length(state.regional_coherence) == state.n_regions &&
           all(isfinite, state.phi_planetary) &&
           all(value -> isfinite(value) && 0.0 <= value <= 1.0, state.regional_coherence) &&
           isfinite(state.t)
end

function encode(state::L6_PlanetaryAdapterState)::Matrix{UInt8}
    probabilities = clamp.(state.regional_coherence, 0.0, 1.0)
    return UInt8.(rand(state.rng, state.n_regions, state.bitstream_length) .< probabilities)
end

function project_inputs(
    inputs::Union{Nothing,AbstractMatrix{<:Real}},
    n_regions::Int,
    bitstream_length::Int,
)::Vector{Float64}
    inputs === nothing && return zeros(Float64, n_regions)
    n_rows, n_cols = size(inputs)
    n_rows > 0 || throw(ArgumentError("inputs must contain at least one row."))
    n_cols == bitstream_length ||
        throw(ArgumentError("inputs bitstream_length must match adapter parameters."))
    all(isfinite, inputs) || throw(ArgumentError("inputs must contain only finite values."))

    raw = vec(mean(Float64.(inputs); dims = 2))
    length(raw) == n_regions && return raw
    return fill(mean(raw), n_regions)
end

function gaia_kernel(
    phi::AbstractVector{<:Real},
    sync_inputs::AbstractVector{<:Real},
    alpha::Real,
    freq::Real,
    q_factor::Real,
    p_percolation::Real,
    t::Real,
    dt::Real,
)::Tuple{Vector{Float64},Vector{Float64}}
    length(phi) > 0 && length(phi) == length(sync_inputs) ||
        throw(ArgumentError("phi and sync input dimensions must be equal and non-empty."))
    all(isfinite, phi) && all(isfinite, sync_inputs) ||
        throw(ArgumentError("phi and sync inputs must be finite."))
    validate_params(length(phi), 1, freq, q_factor, alpha, p_percolation)
    isfinite(t) || throw(ArgumentError("time must be finite."))
    isfinite(dt) && dt > 0.0 || throw(ArgumentError("dt must be finite and positive."))

    bounded_sync = clamp.(Float64.(sync_inputs), 0.0, 1.0)
    order_parameter = clamp(mean(bounded_sync), 0.0, 1.0)
    driver = cos(2.0 * pi * Float64(freq) * Float64(t))
    superradiant_gain = 1.0 + Float64(q_factor) * order_parameter^2
    percolation_gate =
        1.0 / (1.0 + exp(-Float64(q_factor) * (order_parameter - Float64(p_percolation))))

    phi_next = Vector{Float64}(undef, length(phi))
    coherence_next = Vector{Float64}(undef, length(phi))
    for index in eachindex(phi_next)
        d_phi =
            Float64(alpha) * bounded_sync[index] * superradiant_gain * driver -
            0.05 * Float64(phi[index])
        phi_next[index] = Float64(phi[index]) + d_phi * Float64(dt)
        local_field_activation = 1.0 - exp(-Float64(q_factor) * abs(phi_next[index]))
        coherence_next[index] = clamp(percolation_gate * local_field_activation, 0.0, 1.0)
    end
    return phi_next, coherence_next
end

function step_jax(
    state::L6_PlanetaryAdapterState,
    dt::Real,
    inputs::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
)::Matrix{UInt8}
    isfinite(dt) && dt > 0.0 || throw(ArgumentError("dt must be finite and positive."))
    sync_drive = project_inputs(inputs, state.n_regions, state.bitstream_length)
    state.t += Float64(dt)
    state.phi_planetary, state.regional_coherence = gaia_kernel(
        state.phi_planetary,
        sync_drive,
        state.alpha_gaia,
        state.f_schumann,
        state.q_factor,
        state.p_percolation,
        state.t,
        dt,
    )
    return encode(state)
end

function decode(bitstreams::AbstractMatrix{<:Real})::Dict{String,Float64}
    length(bitstreams) > 0 || throw(ArgumentError("bitstreams must be a non-empty matrix."))
    return Dict("global_coherence_index" => mean(Float64.(bitstreams)))
end

function get_metrics(state::L6_PlanetaryAdapterState)::Dict{String,Float64}
    return Dict(
        "gaia_potential" => mean(state.phi_planetary),
        "percolation_index" => mean(state.regional_coherence),
        "schumann_phase" => mod(state.t * state.f_schumann, 1.0),
    )
end

end # module L6PlanAccel
