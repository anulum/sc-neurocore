# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for hdc/base

module BaseAccel

using Statistics, LinearAlgebra

mutable struct AssociativeMemoryState
    dim::Float64
    memory::Float64
end

function AssociativeMemoryState()
    AssociativeMemoryState(10000.0, 0.0)
end

function generate_random_vector(s::AssociativeMemoryState)
    # We use {0, 1} for compatibility with our SC
    return np.random.randint(0, 2, s.dim).astype(np.uint8)
end

function bind(s::AssociativeMemoryState, v1, v2)
    return np.bitwise_xor(v1, v2)
end

function bundle(s::AssociativeMemoryState, vectors)
    if ! vectors
        return zeros(s.dim, dtype=np.uint8)
    # Sum columns
    sum_vec = sum(vectors, axis=0)
    threshold = length(vectors) / 2.0
    return (sum_vec > threshold).astype(np.uint8)
end

function permute(s::AssociativeMemoryState, v, shifts)
    return np.roll(v, shifts)
end

function store(s::AssociativeMemoryState, label, vector)
    s.memory[label] = vector
end

function query(s::AssociativeMemoryState, query_vec)
    best_label = nothing
    min_dist = float("inf")
    for label, mem_vec in s.memory.items()
        # Hamming distance = count(XOR)
        dist = np.count_nonzero(np.bitwise_xor(query_vec, mem_vec))
        if dist < min_dist
            min_dist = dist  # type: ignore[assignment]
            best_label = label
    return best_label
end

end # module BaseAccel
