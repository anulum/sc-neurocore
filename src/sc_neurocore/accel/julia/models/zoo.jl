# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for models/zoo

module ZooAccel

using Statistics, LinearAlgebra

mutable struct SCKeywordSpotterState
    conv::Float64
    dense::Float64
    classifier::Float64
end

function SCKeywordSpotterState()
    SCKeywordSpotterState(0.0, 0.0, 0.0)
end

function forward(s::SCKeywordSpotterState, image, Any])
    # Ensure correct shape (1, 28, 28)
    if image.ndim == 2
        image = image[nothing, :, :]
    # 1. Conv
    features = s.conv.forward(image)
    # Flatten
    flat_features = features.flatten()
    # 2. Dense
    # Vectorized layer expects list/array of floats as probabilities
    # We need to map the conv output (accumulated bit counts) to probabilities [0,1]
    # Conv output is roughly sum of bits. Max bits = kernel_size^2 * length?
    # Let's normalize assuming max overlap
    norm_factor = (3 * 3) * 256
    flat_probs = flat_features / norm_factor
    flat_probs = clamp(flat_probs, 0, 1)
    outputs = s.dense.forward(flat_probs)  # type: ignore[arg-type]
    # Argmax
    return int(argmax(outputs))
end

function predict(s::SCKeywordSpotterState, mfcc_features, Any])
    return int(argmax(s.classifier.forward(mfcc_features)))
end

end # module ZooAccel
