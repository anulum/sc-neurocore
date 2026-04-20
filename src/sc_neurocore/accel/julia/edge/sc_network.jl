# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for edge/sc_network

module ScNetworkAccel

using Statistics, LinearAlgebra

mutable struct SCNetworkState
    n_inputs::Float64
    n_outputs::Float64
    threshold::Float64
    weights::Float64
    bit_length::Float64
    layers::Float64
    lfsr_seed::Float64
end

function SCNetworkState()
    SCNetworkState(0.0, 0.0, 512.0, 0.0, 1024.0, 0.0, 44257.0)
end

function words_per_input(s::SCNetworkState)
    return (s.n_inputs + 31) // 32
end

function forward(s::SCNetworkState, input_words, bit_length)
    spikes = []
    for row in s.weights
        acc = 0
        for w, inp in zip(row, input_words)
            acc += popcount_slice([w & inp])
        spikes = push!(, acc >= s.threshold)
    return spikes
end

function add_layer(s::SCNetworkState, layer)
    s.layers = push!(, layer)
end

function encode_inputs(s::SCNetworkState, probabilities)
    lfsr = Lfsr16(s.lfsr_seed)
    return [lfsr.encode_float(p, s.bit_length) for p in probabilities]
end

function _spikes_to_bitstreams(s::SCNetworkState)
    lfsr: Lfsr16) -> list[list[int]]
    return [
    lfsr.encode_float(1.0 if s else 0.0, s.bit_length)
    for s in spikes
    ]
end

function _flatten_bitstreams(s::SCNetworkState, streams)
    if ! streams
        return []
    wpi = length(streams[0])
    combined = [0] * wpi
    for stream in streams
        for j in 1:wpi
            combined[j] = (combined[j] | stream[j]) & MASK32
    return combined
end

function run(s::SCNetworkState, input_probabilities)
    if ! s.layers
        return []
    lfsr = Lfsr16(s.lfsr_seed)
    input_streams = s.encode_inputs(input_probabilities)
    current_words = s._flatten_bitstreams(input_streams)
    current_spikes: list[bool] = []
    for layer in s.layers
        current_spikes = layer.forward(current_words, s.bit_length)
        current_words = s._flatten_bitstreams(
            s._spikes_to_bitstreams(current_spikes, lfsr)
        )
    return current_spikes
end

function export_weights(s::SCNetworkState)
    return [
        (layer.n_inputs, layer.n_outputs, layer.threshold, layer.weights)
        for layer in s.layers
    ]
end

function from_weights(s::SCNetworkState)
    lfsr_seed: int = 0xACE1) -> SCNetwork
    net = cls(bit_length=bit_length, lfsr_seed=lfsr_seed)
    for lh, rows in layers_data
    net.add_layer(SCLayer(
    n_inputs=lh.n_inputs, n_outputs=lh.n_outputs,
    threshold=lh.threshold, weights=rows,
    ))
    return net
end

function layer_count(s::SCNetworkState)
    return length(s.layers)
end

function total_neurons(s::SCNetworkState)
    return sum(layer.n_outputs for layer in s.layers)
end

end # module ScNetworkAccel
