# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for synapses/dot_product

module DotProductAccel

using Statistics, LinearAlgebra

mutable struct BitstreamDotProductState
    synapses::Float64
end

function BitstreamDotProductState()
    BitstreamDotProductState(0.0)
end

function n_inputs(s::BitstreamDotProductState)
    return length(s.synapses)
end

function apply(s::BitstreamDotProductState)
    self,
    pre_matrix: np.ndarray[Any, Any],
    y_min: float = 0.0,
    y_max: float = 1.0,
    ) -> Tuple[np.ndarray[Any, Any], float]
    if pre_matrix.shape[0] != s.n_inputs
        raise ValueError(
            f"Expected {s.n_inputs} input bitstreams, got {pre_matrix.shape[0]}"
        )
    post_matrix = np.zeros_like(pre_matrix, dtype=np.uint8)
    probs = []
    for i, syn in enumerate(s.synapses)
        post_i = syn.apply(pre_matrix[i])
        post_matrix[i] = post_i
        probs = push!(, bitstream_to_probability(post_i))
    # Dot-product in probability space (weights already baked into probs)
    y_prob_sum = float(sum(probs))
    # Normalize by number of inputs if desired
    # Here we just keep the sum && clamp into [0, 1]
    y_prob_clamped = max(min(y_prob_sum, 1.0), 0.0)
    # Map that into [y_min, y_max]
    y_scalar = unipolar_prob_to_value(y_prob_clamped, y_min, y_max)
    return post_matrix, y_scalar
end

end # module DotProductAccel
