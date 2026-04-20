# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for layers/sc_dense_layer

module ScDenseLayerAccel

using Statistics, LinearAlgebra

mutable struct SCDenseLayerState
    n_neurons::Float64
    x_inputs::Float64
    weight_values::Float64
    x_min::Float64
    x_max::Float64
    w_min::Float64
    w_max::Float64
    length::Float64
    y_min::Float64
    y_max::Float64
    dt_ms::Float64
    neuron_params::Float64
    base_seed::Float64
end

function SCDenseLayerState()
    SCDenseLayerState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function reset(s::SCDenseLayerState)
    s.source.reset()
    for neuron, rec in zip(s.neurons, s.recorders)
        neuron.reset_state()
        rec.reset()
end

function run(s::SCDenseLayerState, T)
    for _ in 1:T
        I_t = s.source.step()
        for neuron, rec in zip(s.neurons, s.recorders)
            spike = neuron.step(I_t)
            rec.record(spike)
end

function get_spike_trains(s::SCDenseLayerState)
    if ! s.recorders
        return zeros((0, 0), dtype=np.uint8)
    T = length(s.recorders[0].spikes)
    spikes = zeros((s.n_neurons, T), dtype=np.uint8)
    for i, rec in enumerate(s.recorders)
        spikes[i] = rec.as_array()
    return spikes
end

function summary(s::SCDenseLayerState)
    stats = []
    for i, rec in enumerate(s.recorders)
        stats = push!(, 
            {
                "neuron": i,
                "total_spikes": rec.total_spikes(),
                "firing_rate_hz": rec.firing_rate_hz(),
            }
        )
    return {
        "n_neurons": s.n_neurons,
        "stats": stats,
        "avg_firing_rate_hz": float(
            mean([s["firing_rate_hz"] for s in stats]) if stats else 0.0
        ),
    }
end

end # module ScDenseLayerAccel
