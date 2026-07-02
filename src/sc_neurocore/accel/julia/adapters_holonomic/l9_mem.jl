# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia safety mirror for adapters_holonomic/l9_mem

module L9MemAccel

using Random
using Statistics

export L9_MemoryAdapterState,
    decode,
    encode,
    get_metrics,
    project_inputs,
    step_jax,
    validate_l9_mem,
    validate_params

mutable struct L9_MemoryAdapterState
    n_memory_slots::Int
    bitstream_length::Int
    retrieval_gain::Float64
    weak_measurement_strength::Float64
    temporal_window::Int
    rng::MersenneTwister
    imprints_psi::Matrix{UInt8}
    retrieval_phi::Matrix{UInt8}
    current_slot::Int
end

function L9_MemoryAdapterState(;
    n_memory_slots::Int = 64,
    bitstream_length::Int = 1024,
    retrieval_gain::Real = 0.8,
    weak_measurement_strength::Real = 0.1,
    temporal_window::Int = 100,
    seed::Int = 49,
)::L9_MemoryAdapterState
    gain = Float64(retrieval_gain)
    strength = Float64(weak_measurement_strength)
    validate_params(n_memory_slots, bitstream_length, gain, strength, temporal_window)
    return L9_MemoryAdapterState(
        n_memory_slots,
        bitstream_length,
        gain,
        strength,
        temporal_window,
        MersenneTwister(seed),
        zeros(UInt8, n_memory_slots, bitstream_length),
        zeros(UInt8, n_memory_slots, bitstream_length),
        0,
    )
end

function validate_params(
    n_memory_slots::Int,
    bitstream_length::Int,
    retrieval_gain::Real,
    weak_measurement_strength::Real,
    temporal_window::Int,
)::Nothing
    n_memory_slots > 0 || throw(ArgumentError("n_memory_slots must be positive."))
    bitstream_length > 0 || throw(ArgumentError("bitstream_length must be positive."))
    temporal_window > 0 || throw(ArgumentError("temporal_window must be positive."))
    isfinite(retrieval_gain) && retrieval_gain >= 0.0 ||
        throw(ArgumentError("retrieval_gain must be finite and non-negative."))
    isfinite(weak_measurement_strength) &&
        0.0 <= weak_measurement_strength <= 1.0 ||
        throw(ArgumentError("weak_measurement_strength must be finite and in [0, 1]."))
    return nothing
end

function validate_l9_mem(state::L9_MemoryAdapterState)::Bool
    try
        validate_params(
            state.n_memory_slots,
            state.bitstream_length,
            state.retrieval_gain,
            state.weak_measurement_strength,
            state.temporal_window,
        )
    catch
        return false
    end
    if !(0 <= state.current_slot < state.n_memory_slots)
        return false
    end
    expected_shape = (state.n_memory_slots, state.bitstream_length)
    return size(state.imprints_psi) == expected_shape &&
           size(state.retrieval_phi) == expected_shape &&
           all(bit -> bit <= 0x01, state.imprints_psi) &&
           all(bit -> bit <= 0x01, state.retrieval_phi)
end

function encode(state::L9_MemoryAdapterState)::Vector{UInt8}
    psi_float = Float64.(state.imprints_psi)
    phi_float = Float64.(state.retrieval_phi)
    overlap = vec(mean(psi_float .* phi_float; dims = 2))
    retrieval_prob = clamp(sum(overlap) * state.retrieval_gain, 0.0, 1.0)
    return UInt8.(rand(state.rng, state.bitstream_length) .< retrieval_prob)
end

function _tsvf_kernel(
    psi::Matrix{UInt8},
    phi::Matrix{UInt8},
    inputs::Matrix{Float64},
    strength::Float64,
)::Tuple{Matrix{UInt8},Matrix{UInt8}}
    psi_next = copy(psi)
    psi_next[inputs .> 0.5] .= 0x01
    phi_next = copy(phi)
    phi_next[abs.(Float64.(psi_next) .- 0.5) .> strength] .= 0x01
    return psi_next, phi_next
end

function project_inputs(
    inputs::AbstractMatrix{<:Real},
    n_memory_slots::Int,
    bitstream_length::Int,
)::Matrix{Float64}
    n_rows, n_cols = size(inputs)
    n_rows > 0 || throw(ArgumentError("inputs must contain at least one row."))
    n_cols == bitstream_length ||
        throw(ArgumentError("inputs bitstream_length must match adapter parameters."))
    all(isfinite, inputs) || throw(ArgumentError("inputs must contain only finite values."))

    indices = [((slot - 1) % n_rows) + 1 for slot in 1:n_memory_slots]
    return Float64.(inputs[indices, :])
end

function step_jax(
    state::L9_MemoryAdapterState,
    dt::Real,
    inputs::Union{Nothing,AbstractMatrix{<:Real}} = nothing,
)::Vector{UInt8}
    isfinite(dt) && dt > 0.0 || throw(ArgumentError("dt must be finite and positive."))
    if inputs !== nothing
        mapped_inputs = project_inputs(inputs, state.n_memory_slots, state.bitstream_length)
        state.imprints_psi, state.retrieval_phi = _tsvf_kernel(
            state.imprints_psi,
            state.retrieval_phi,
            mapped_inputs,
            state.weak_measurement_strength,
        )
    end
    return encode(state)
end

function decode(bitstreams::AbstractVector{<:Real})::Dict{String,Float64}
    length(bitstreams) > 0 || throw(ArgumentError("bitstreams must not be empty."))
    return Dict("memory_retrieval_r9" => mean(Float64.(bitstreams)))
end

function get_metrics(state::L9_MemoryAdapterState)::Dict{String,Float64}
    overlap = mean(Float64.(state.imprints_psi) .* Float64.(state.retrieval_phi))
    imprint_density = mean(Float64.(state.imprints_psi))
    return Dict("holographic_overlap" => overlap, "imprint_density" => imprint_density)
end

end # module L9MemAccel
