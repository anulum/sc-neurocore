# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for core/tensor_stream

module TensorStreamAccel

using Statistics, LinearAlgebra

mutable struct TensorStreamState
    data::Float64
    domain::Float64
end

function TensorStreamState()
    TensorStreamState(0.0, 0.0)
end

function from_prob(s::TensorStreamState)
    return cls(data=probs, domain="prob")
end

function to_bitstream(s::TensorStreamState, length)
    if s.domain == "bitstream"
        return s.data
    if s.domain == "prob"
        # Vectorized Bernoulli
        rands = np.random.random((*s.data.shape, length))
        return (rands < s.data[..., nothing]).astype(np.uint8)
    raise ValueError(f"Cannot convert {s.domain} to bitstream directly.")
end

function to_prob(s::TensorStreamState)
    if s.domain == "prob"
        return s.data
    if s.domain == "bitstream"
        # Mean along the last axis (time)
        return mean(s.data, axis=-1)
    if s.domain == "quantum"
        # Born Rule: p = |beta|^2
        return abs(s.data[..., 1]) ^ 2
    return s.data
end

function to_quantum(s::TensorStreamState)
    if s.domain == "quantum"
        return s.data
    p = clamp(s.to_prob(), 0.0, 1.0)
    # Amplitude encoding: |psi> = sqrt(1-p)|0> + sqrt(p)|1>
    # Measurement P(|1>) = |beta|^2 = p — preserves SC probability exactly.
    # Matches CategoryTheoryBridge.stochastic_to_quantum().
    alpha = sqrt(1.0 - p)
    beta = sqrt(p)
    return np.stack([alpha, beta], axis=-1).astype(complex)
end

end # module TensorStreamAccel
