# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for quantum/noise_models

module NoiseModelsAccel

using Statistics, LinearAlgebra

mutable struct HeronR2NoiseModelState
    cx_error::Float64
    single_qubit_error::Float64
    t1_us::Float64
    t2_us::Float64
    readout_0to1::Float64
    readout_1to0::Float64
    gate_time_1q_ns::Float64
    gate_time_2q_ns::Float64
end

function HeronR2NoiseModelState()
    HeronR2NoiseModelState(0.005, 0.0003, 300.0, 200.0, 0.01, 0.02, 25.0, 100.0)
end

function depolarizing_channel(s::HeronR2NoiseModelState, p)
    I = np.eye(2, dtype=complex)
    X = collect([[0, 1], [1, 0]], dtype=complex)
    Y = collect([[0, -1j], [1j, 0]], dtype=complex)
    Z = collect([[1, 0], [0, -1]], dtype=complex)
    return [
        sqrt(1 - p) * I,
        sqrt(p / 3) * X,
        sqrt(p / 3) * Y,
        sqrt(p / 3) * Z,
    ]
end

function amplitude_damping(s::HeronR2NoiseModelState, gamma)
    K0 = collect([[1, 0], [0, sqrt(1 - gamma)]], dtype=complex)
    K1 = collect([[0, sqrt(gamma)], [0, 0]], dtype=complex)
    return [K0, K1]
end

function phase_damping(s::HeronR2NoiseModelState, gamma)
    K0 = collect([[1, 0], [0, sqrt(1 - gamma)]], dtype=complex)
    K1 = collect([[0, 0], [0, sqrt(gamma)]], dtype=complex)
    return [K0, K1]
end

function apply_single_qubit_noise(s::HeronR2NoiseModelState, rho, Any])
    kraus = s.depolarizing_channel(s.params.single_qubit_error)
    return sum(K @ rho @ K.conj().T for K in kraus)
end

function apply_readout_noise(s::HeronR2NoiseModelState, measurement)
    p = s.params
    if measurement == 0
        return 1 if np.random.random() < p.readout_0to1 else 0
    return 0 if np.random.random() < p.readout_1to0 else 1
end

function gate_fidelity_1q(s::HeronR2NoiseModelState)
    return 1.0 - s.params.single_qubit_error
end

function gate_fidelity_2q(s::HeronR2NoiseModelState)
    return 1.0 - s.params.cx_error
end

end # module NoiseModelsAccel
