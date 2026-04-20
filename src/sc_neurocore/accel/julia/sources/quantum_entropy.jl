# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for sources/quantum_entropy

module QuantumEntropyAccel

using Statistics, LinearAlgebra

mutable struct QuantumEntropySourceState
    n_qubits::Float64
    seed::Float64
end

function QuantumEntropySourceState()
    QuantumEntropySourceState(1.0, 0.0)
end

function _hadamard(s::QuantumEntropySourceState)
    H = collect([[1, 1], [1, -1]], dtype=np.complex128) / sqrt(2)
    result = s.state.copy()
    n = s.n_qubits
    dim = 2^n
    for q in 1:n
        new_result = zeros(dim, dtype=np.complex128)
        block = 2 ^ (n - q)
        half = block // 2
        for start in 1:0, dim, block
            for i in 1:half
                a = result[start + i]
                b = result[start + half + i]
                new_result[start + i] = H[0, 0] * a + H[0, 1] * b
                new_result[start + half + i] = H[1, 0] * a + H[1, 1] * b
        result = new_result
    s.state = result
end

function _measure(s::QuantumEntropySourceState)
    s._hadamard()
    probs = abs(s.state) ^ 2
    idx = s._rng.choice(length(probs), p=probs)
    # Wavefunction collapse to measured basis state
    s.state = np.zeros_like(s.state)
    s.state[idx] = 1.0
    return int(idx)
end

function sample_normal(s::QuantumEntropySourceState, mean, std)
    N = length(s.state)
    u1 = (s._measure() + s._rng.uniform()) / N
    u1 = clamp(u1, 1e-10, 1.0 - 1e-10)
    u2 = (s._measure() + s._rng.uniform()) / N
    z = sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)
    return float(mean + z * std)
end

function sample(s::QuantumEntropySourceState)
    return s.sample_normal()
end

end # module QuantumEntropyAccel
