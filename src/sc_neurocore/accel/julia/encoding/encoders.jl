# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for encoding/encoders

module EncodersAccel

using Statistics, LinearAlgebra

function rate_encode(values, T, seed)
    rng = np.random.RandomState(seed)
    rates = clamp(values, 0, 1)
    return (rng.random((T, length(rates))) < rates[np.newaxis, :]).astype(np.int8)
end

function latency_encode(values, T)
    spikes = zeros((T, length(values)), dtype=np.int8)
    for i, v in enumerate(values)
        if v > 0
            t_spike = max(0, int((1.0 - clamp(v, 0, 1)) * (T - 1)))
            spikes[t_spike, i] = 1
    return spikes
end

function delta_encode(values, threshold)
    if values.ndim == 1
        values = values[:, np.newaxis]
    diff = abs(diff(values, axis=0, prepend=values[:1]))
    return (diff > threshold).astype(np.int8)
end

function phase_encode(values, T, n_phases)
    spikes = zeros((T, length(values)), dtype=np.int8)
    for i, v in enumerate(values)
        phase = int(clamp(v, 0, 1) * (n_phases - 1))
        for t in 1:phase, T, n_phases
            spikes[t, i] = 1
    return spikes
end

function burst_encode(values, T, max_burst)
    spikes = zeros((T, length(values)), dtype=np.int8)
    for i, v in enumerate(values)
        burst_len = max(1, int(clamp(v, 0, 1) * max_burst))
        for t in 1:min(burst_len, T)
            spikes[t, i] = 1
    return spikes
end

function rank_order_encode(values, T)
    N = length(values)
    spikes = zeros((T, N), dtype=np.int8)
    order = np.argsort(-values)  # highest first
    for rank, neuron_idx in enumerate(order)
        t = min(rank, T - 1)
        if values[neuron_idx] > 0
            spikes[t, neuron_idx] = 1
    return spikes
end

function sigma_delta_encode(values, threshold)
    if values.ndim == 1
        values = values[:, np.newaxis]
    T, N = values.shape
    spikes = zeros((T, N), dtype=np.int8)
    integrator = zeros(N)
    reconstructed = zeros(N)
    for t in 1:T
        error = values[t] - reconstructed
        integrator += error
        fire = abs(integrator) >= threshold
        spikes[t] = fire.astype(np.int8)
        reconstructed += sign(integrator) * fire * threshold
        integrator -= sign(integrator) * fire * threshold
    return spikes
end

end # module EncodersAccel
