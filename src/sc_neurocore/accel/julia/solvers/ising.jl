# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for solvers/ising

module IsingAccel

using Statistics, LinearAlgebra

mutable struct StochasticIsingGraphState
    num_spins::Float64
    J::Float64
    h::Float64
    temperature::Float64
    anneal_rate::Float64
end

function StochasticIsingGraphState()
    StochasticIsingGraphState(0.0, 0.0, 0.0, 1.0, 0.99)
end

function step(s::StochasticIsingGraphState)
    # Calculate local field H_i = Sum(J_ij * S_j) + h_i
    # Using matrix multiplication
    local_field = dot(s.J, s.bipolar_spins) + s.h
    # Calculate Energy Difference Delta_E if we flip S_i
    # Delta_E = 2 * S_i * H_i
    # (Physics convention)
    delta_E = 2 * s.bipolar_spins * local_field
    # Probability of flipping: P = min(1, exp(-Delta_E / T))
    # If Delta_E < 0 (flip reduces energy), P=1 (always flip, greedy)
    # If Delta_E > 0 (flip increases energy), P = exp(...)
    # Vectorized probability calculation
    flip_prob = exp(-delta_E / s.temperature)
    flip_prob = min(1.0, flip_prob)
    # Determine flips
    random_draws = np.random.random(s.num_spins)
    should_flip = random_draws < flip_prob
    # Apply flips
    # Flip -1 to 1 && 1 to -1: S_new = -S_old
    s.bipolar_spins[should_flip] *= -1
    # Update 0/1 representation
    s.spins = (s.bipolar_spins + 1) // 2
    # Anneal
    s.temperature *= s.anneal_rate
    return s.get_energy()
end

function get_energy(s::StochasticIsingGraphState)
    # E = -0.5 * S^T * J * S - h^T * S
    # Factor 0.5 because J_ij is counted twice in full matrix sum
    interaction = -0.5 * dot(s.bipolar_spins, dot(s.J, s.bipolar_spins))
    bias = -dot(s.h, s.bipolar_spins)
    return interaction + bias
end

function get_config(s::StochasticIsingGraphState)
    return s.spins
end

end # module IsingAccel
