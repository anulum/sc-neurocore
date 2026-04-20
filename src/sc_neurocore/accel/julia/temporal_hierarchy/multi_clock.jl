# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for temporal_hierarchy/multi_clock

module MultiClockAccel

using Statistics, LinearAlgebra

mutable struct MultiClockSNNState
    name::Float64
    tick_interval::Float64
    layers::Float64
    n_inputs::Float64
    n_neurons::Float64
    threshold::Float64
    tau::Float64
    W::Float64
    _traces::Float64
    _v::Float64
    layer_names::Float64
    clock_intervals::Float64
    _step_count::Float64
end

function MultiClockSNNState()
    MultiClockSNNState(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
end

function step(s::MultiClockSNNState, x, dt)
    decay = exp(-dt / s.tau)
    s._traces = decay * s._traces + x[np.newaxis, :]
    current = (s.W * s._traces).sum(axis=1)
    s._v += current
    spikes = (s._v >= s.threshold).astype(np.float64)
    s._v -= spikes * s.threshold
    return spikes
end

function reset(s::MultiClockSNNState)
    s._traces = zeros((s.n_neurons, s.n_inputs))
    s._v = zeros(s.n_neurons)
end

function tau_stats(s::MultiClockSNNState)
    return {
        "mean": float(s.tau.mean()),
        "std": float(s.tau.std()),
        "min": float(s.tau.min()),
        "max": float(s.tau.max()),
        "median": float(np.median(s.tau)),
    }
end

function step(s::MultiClockSNNState, x, dt)
    s._step_count += 1
    h = x.astype(np.float64)
    for i, (layer, interval) in enumerate(zip(s.layers, s.clock_intervals))
        if s._step_count % interval == 0
            spikes = layer.step(h, dt=dt * interval)
            s._last_outputs[i] = spikes
        h = s._last_outputs[i]
    return h
end

function run(s::MultiClockSNNState, inputs, dt)
    s.reset()
    T = inputs.shape[0]
    n_out = s.layers[-1].n_neurons
    outputs = zeros((T, n_out))
    for t in 1:T
        outputs[t] = s.step(inputs[t], dt)
    return outputs
end

function reset(s::MultiClockSNNState)
    s._step_count = 0
    for i, layer in enumerate(s.layers)
        layer.reset()
        s._last_outputs[i] = zeros(layer.n_neurons)
end

end # module MultiClockAccel
