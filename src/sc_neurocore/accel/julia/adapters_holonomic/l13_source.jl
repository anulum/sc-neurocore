# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia safety mirror for adapters_holonomic/l13_source

module L13SourceAccel

using Random
using Statistics

export L13_SourceAdapterState,
    decode,
    encode,
    get_metrics,
    project_feedback,
    step_jax,
    vacuum_lattice_kernel,
    validate_l13_source,
    validate_params

mutable struct L13_SourceAdapterState
    n_vacuum_nodes::Int
    bitstream_length::Int
    j_primordial_coupling::Float64
    h_potential_bias::Float64
    lambda_scission::Float64
    rng::MersenneTwister
    vacuum_state::Vector{Float64}
    fim_density::Vector{Float64}
end

function L13_SourceAdapterState(;
    n_vacuum_nodes::Int = 256,
    bitstream_length::Int = 1024,
    j_primordial_coupling::Real = 1.0,
    h_potential_bias::Real = 0.01,
    lambda_scission::Real = 0.1,
    seed::Int = 413,
)::L13_SourceAdapterState
    coupling = Float64(j_primordial_coupling)
    bias = Float64(h_potential_bias)
    scission = Float64(lambda_scission)
    validate_params(n_vacuum_nodes, bitstream_length, coupling, bias, scission)

    rng = MersenneTwister(seed)
    vacuum_state = fill(0.5, n_vacuum_nodes)
    if scission > 0.0
        amplitude = min(scission, 1.0) * 0.02
        vacuum_state = clamp.(vacuum_state .+ (rand(rng, n_vacuum_nodes) .- 0.5) .* amplitude, 0.0, 1.0)
    end
    return L13_SourceAdapterState(
        n_vacuum_nodes,
        bitstream_length,
        coupling,
        bias,
        scission,
        rng,
        vacuum_state,
        zeros(Float64, n_vacuum_nodes),
    )
end

function validate_params(
    n_vacuum_nodes::Int,
    bitstream_length::Int,
    j_primordial_coupling::Real,
    h_potential_bias::Real,
    lambda_scission::Real,
)::Nothing
    n_vacuum_nodes > 0 || throw(ArgumentError("n_vacuum_nodes must be positive."))
    bitstream_length > 0 || throw(ArgumentError("bitstream_length must be positive."))
    isfinite(j_primordial_coupling) || throw(ArgumentError("j_primordial_coupling must be finite."))
    isfinite(h_potential_bias) || throw(ArgumentError("h_potential_bias must be finite."))
    isfinite(lambda_scission) && lambda_scission >= 0.0 ||
        throw(ArgumentError("lambda_scission must be finite and non-negative."))
    return nothing
end

function validate_l13_source(state::L13_SourceAdapterState)::Bool
    try
        validate_params(
            state.n_vacuum_nodes,
            state.bitstream_length,
            state.j_primordial_coupling,
            state.h_potential_bias,
            state.lambda_scission,
        )
    catch
        return false
    end
    return length(state.vacuum_state) == state.n_vacuum_nodes &&
           length(state.fim_density) == state.n_vacuum_nodes &&
           all(value -> isfinite(value) && 0.0 <= value <= 1.0, state.vacuum_state) &&
           all(isfinite, state.fim_density)
end

function encode(state::L13_SourceAdapterState)::Matrix{UInt8}
    probabilities = clamp.(state.vacuum_state, 0.0, 1.0)
    return UInt8.(rand(state.rng, state.n_vacuum_nodes, state.bitstream_length) .< probabilities)
end

function project_feedback(
    inputs::Union{Nothing,Real,AbstractVector{<:Real},AbstractMatrix{<:Real}},
    n_vacuum_nodes::Int,
)::Vector{Float64}
    inputs === nothing && return zeros(Float64, n_vacuum_nodes)
    if inputs isa Real
        isfinite(inputs) || throw(ArgumentError("inputs must contain only finite values."))
        return fill(clamp(2.0 * Float64(inputs) - 1.0, -1.0, 1.0), n_vacuum_nodes)
    end

    if inputs isa AbstractVector
        length(inputs) > 0 || throw(ArgumentError("inputs must contain at least one value."))
        all(isfinite, inputs) || throw(ArgumentError("inputs must contain only finite values."))
        raw = Float64.(inputs)
    else
        n_rows, n_cols = size(inputs)
        n_rows > 0 || throw(ArgumentError("inputs must contain at least one row."))
        n_cols > 0 || throw(ArgumentError("inputs must contain at least one column."))
        all(isfinite, inputs) || throw(ArgumentError("inputs must contain only finite values."))
        raw = vec(mean(Float64.(inputs); dims = 2))
    end

    projected = length(raw) == n_vacuum_nodes ? raw : fill(mean(raw), n_vacuum_nodes)
    return clamp.(2.0 .* projected .- 1.0, -1.0, 1.0)
end

function vacuum_lattice_kernel(
    state::AbstractVector{<:Real},
    coupling::Real,
    bias::Real,
    scission_rate::Real,
    feedback_drive::AbstractVector{<:Real},
    dt::Real,
)::Vector{Float64}
    length(state) > 0 && length(state) == length(feedback_drive) ||
        throw(ArgumentError("state and feedback dimensions must be equal and non-empty."))
    all(isfinite, state) && all(isfinite, feedback_drive) ||
        throw(ArgumentError("state and feedback must be finite."))
    validate_params(length(state), 1, coupling, bias, scission_rate)
    isfinite(dt) && dt > 0.0 || throw(ArgumentError("dt must be finite and positive."))

    spin = 2.0 .* clamp.(Float64.(state), 0.0, 1.0) .- 1.0
    out = Vector{Float64}(undef, length(spin))
    for index in eachindex(spin)
        left = index == firstindex(spin) ? lastindex(spin) : index - 1
        right = index == lastindex(spin) ? firstindex(spin) : index + 1
        neighbour_field = 0.5 * (spin[left] + spin[right])
        hamiltonian_drive =
            Float64(coupling) * neighbour_field +
            Float64(bias) +
            0.25 * clamp(Float64(feedback_drive[index]), -1.0, 1.0)
        scission_drive = Float64(scission_rate) * (spin[index] - spin[index]^3)
        relaxation = -0.05 * spin[index]
        spin_next = spin[index] + (hamiltonian_drive + scission_drive + relaxation) * Float64(dt)
        out[index] = clamp(0.5 * (spin_next + 1.0), 0.0, 1.0)
    end
    return out
end

function step_jax(
    state::L13_SourceAdapterState,
    dt::Real,
    inputs::Union{Nothing,Real,AbstractVector{<:Real},AbstractMatrix{<:Real}} = nothing,
)::Matrix{UInt8}
    isfinite(dt) && dt > 0.0 || throw(ArgumentError("dt must be finite and positive."))
    feedback_drive = project_feedback(inputs, state.n_vacuum_nodes)
    previous = copy(state.vacuum_state)
    state.vacuum_state = vacuum_lattice_kernel(
        state.vacuum_state,
        state.j_primordial_coupling,
        state.h_potential_bias,
        state.lambda_scission,
        feedback_drive,
        dt,
    )

    variance = max.(state.vacuum_state .* (1.0 .- state.vacuum_state), 1.0e-6)
    temporal_delta = state.vacuum_state .- previous
    lattice_delta = circshift(state.vacuum_state, -1) .- state.vacuum_state
    instant_fim = (temporal_delta .* temporal_delta .+ lattice_delta .* lattice_delta) ./ variance
    state.fim_density = 0.9 .* state.fim_density .+ 0.1 .* instant_fim
    return encode(state)
end

function decode(bitstreams::AbstractMatrix{<:Real})::Dict{String,Float64}
    length(bitstreams) > 0 || throw(ArgumentError("bitstreams must be a non-empty matrix."))
    all(isfinite, bitstreams) || throw(ArgumentError("bitstreams must contain only finite values."))
    return Dict("source_coherence_r13" => mean(Float64.(bitstreams)))
end

function get_metrics(state::L13_SourceAdapterState)::Dict{String,Float64}
    return Dict(
        "vacuum_potential" => mean(state.vacuum_state),
        "fisher_information_metric" => mean(state.fim_density),
    )
end

end # module L13SourceAccel
