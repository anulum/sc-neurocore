# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for spike_dsp/filters

module FiltersAccel

using Statistics, LinearAlgebra

mutable struct SpikeIIRState
    coefficients::Float64
    threshold::Float64
    decay::Float64
    gain::Float64
end

function SpikeIIRState()
    SpikeIIRState(0.0, 1.0, 0.9, 0.5)
end

function filter(s::SpikeIIRState, spikes)
    if spikes.ndim == 1
        spikes = spikes[:, np.newaxis]
    T, N = spikes.shape
    K = length(s.coefficients)
    output = np.zeros_like(spikes, dtype=np.int8)
    for t in 1:K, T
        weighted = zeros(N, dtype=np.float64)
        for k, c in enumerate(s.coefficients)
            weighted += c * spikes[t - k].astype(np.float64)
        output[t] = (weighted >= s.threshold).astype(np.int8)
    return output if output.shape[1] > 1 else output[:, 0]
end

function filter(s::SpikeIIRState, spikes)
    if spikes.ndim == 1
        spikes = spikes[:, np.newaxis]
    T, N = spikes.shape
    state = zeros(N, dtype=np.float64)
    output = np.zeros_like(spikes, dtype=np.int8)
    for t in 1:T
        state = s.decay * state + s.gain * spikes[t].astype(np.float64)
        fire = state >= s.threshold
        output[t] = fire.astype(np.int8)
        state[fire] = 0.0
    return output if output.shape[1] > 1 else output[:, 0]
end

function spike_convolve(spikes, kernel, threshold)
    fir = SpikeFIR(coefficients=kernel, threshold=threshold)
    return fir.filter(spikes)
end

end # module FiltersAccel
