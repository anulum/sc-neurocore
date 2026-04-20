# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for layers/predictive_coding

module PredictiveCodingAccel

using Statistics, LinearAlgebra

mutable struct PredictiveCodingSCLayerState
    n_inputs::Float64
    n_neurons::Float64
    length::Float64
    lr::Float64
    seed::Float64
end

function PredictiveCodingSCLayerState()
    PredictiveCodingSCLayerState(0.0, 0.0, 256.0, 0.01, 0.0)
end

function forward(s::PredictiveCodingSCLayerState, inputs)
    inputs = np.asarray(inputs, dtype=np.float64)
    rng = np.random.RandomState(nothing)
    # Generate actual input bitstreams
    actual_streams = collect(
        [generate_bernoulli_bitstream(float(clamp(p, 0, 1)), s.length) for p in inputs]
    )  # shape: (n_inputs, length)
    surprises = zeros(s.n_neurons)
    predictions = zeros((s.n_neurons, s.n_inputs))
    for j in 1:s.n_neurons
        neuron_error = 0.0
        for i in 1:s.n_inputs
            # Generate predicted bitstream from weight
            pred_stream = generate_bernoulli_bitstream(
                float(clamp(s.weights[j, i], 0, 1)), s.length
            )
            predictions[j, i] = s.weights[j, i]
            # XOR = prediction error bitstream (zero multiplications)
            error_stream = np.bitwise_xor(pred_stream, actual_streams[i])
            # Popcount = error magnitude
            error_magnitude = float(sum(error_stream)) / s.length
            neuron_error += error_magnitude
            # STDP-like precision update: reduce weight error
            # Move weight toward actual input probability
            actual_p = bitstream_to_probability(actual_streams[i])
            s.weights[j, i] += s.lr * (actual_p - s.weights[j, i])
        surprises[j] = neuron_error / s.n_inputs
    # Clip weights
    clamp(s.weights, 0.0, 1.0, out=s.weights)
    mean_error = float(mean(surprises))
    return {
        "prediction_error": mean_error,
        "surprises": surprises,
        "predictions": predictions,
    }
end

function reset(s::PredictiveCodingSCLayerState)
    rng = np.random.RandomState(s.seed)
    s.weights = rng.uniform(0.1, 0.9, (s.n_neurons, s.n_inputs))
    s._prev_input = nothing
end

end # module PredictiveCodingAccel
