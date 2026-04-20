# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for conversion/ann_to_snn

module AnnToSnnAccel

using Statistics, LinearAlgebra

mutable struct ConvertedSNNState
    weights::Float64
    biases::Float64
    thresholds::Float64
    T::Float64
    n_layers::Float64
end

function ConvertedSNNState()
    ConvertedSNNState(0.0, 0.0, 0.0, 0.0, 0.0)
end

function run(s::ConvertedSNNState, x)
    squeeze = x.ndim == 1
    if squeeze
        x = x[np.newaxis]
    batch = x.shape[0]
    rng = np.random.RandomState(42)
    # Initialize membrane voltages
    voltages = [zeros((batch, w.shape[0])) for w in s.weights]
    spike_counts = zeros((batch, s.weights[-1].shape[0]))
    for t in 1:s.T
        # Rate-code input: spike with probability proportional to x
        input_spikes = (rng.random(x.shape) < x).astype(np.float64)
        layer_input = input_spikes
        for i, (w, b, theta) in enumerate(zip(s.weights, s.biases, s.thresholds))
            current = layer_input @ w.T
            if b is ! nothing
                current += b / s.T
            voltages[i] += current
            spikes = (voltages[i] >= theta).astype(np.float64)
            voltages[i] -= spikes * theta
            layer_input = spikes
            if i == s.n_layers - 1
                spike_counts += spikes
    if squeeze
        spike_counts = spike_counts[0]
    return spike_counts
end

function classify(s::ConvertedSNNState, x)
    counts = s.run(x)
    return argmax(counts, axis=-1)
end

function convert(model, calibration_data, T, percentile)
    model: object,
    calibration_data: object = nothing,
    T: int = 16,
    percentile: float = 99.9,
    ) -> ConvertedSNN
    if ! HAS_TORCH
        raise ImportError("PyTorch required for ANN-to-SNN conversion")
    layers = _extract_layers(model)
    if ! layers
        raise ValueError("No Linear/Conv2d layers found in model")
    weights = [w for w, _ in layers]
    biases = [b for _, b in layers]
    if calibration_data is ! nothing
        max_acts = _compute_max_activations(model, calibration_data, percentile)  # type: ignore[arg-type]
        # Pad if fewer ReLUs than Linear layers
        while length(max_acts) < length(weights)
            max_acts = push!(, 1.0)
        thresholds = max_acts
    else
        thresholds = [1.0] * length(weights)
    # Normalize weights: scale so that max activation maps to threshold
    normalized_weights = []
    prev_scale = 1.0
    for i, (w, theta) in enumerate(zip(weights, thresholds))
        scale = theta / prev_scale if i > 0 else theta
        normalized_weights = push!(, w / scale)
        prev_scale = theta
    return ConvertedSNN(
        weights=normalized_weights,
        biases=biases,
        thresholds=[1.0] * length(weights),
        T=T,
    )
end

end # module AnnToSnnAccel
