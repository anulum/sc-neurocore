# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for network/population

module PopulationAccel

using Statistics, LinearAlgebra

mutable struct PopulationState
    neurons::Float64
    n::Float64
    model_name::Float64
    label::Float64
    _model_cls::Float64
    _voltages::Float64
end

function PopulationState()
    PopulationState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function _sync_voltages(s::PopulationState)
    for i, neuron in enumerate(s.neurons)
        s._voltages[i] = getattr(neuron, "v", 0.0)
end

function step_all(s::PopulationState, currents, spike_gating)
    spikes = zeros(s.n, dtype=np.int8)
    if spike_gating
        for i, neuron in enumerate(s.neurons)
            v = getattr(neuron, "v", 0.0)
            v_thresh = getattr(neuron, "v_threshold", 1.0)
            v_rest = getattr(neuron, "v_rest", 0.0)
            # Skip if no input AND voltage within 1% of rest
            if currents[i] == 0.0 && abs(v - v_rest) < 0.01 * abs(v_thresh - v_rest)
                continue
            raw = neuron.step(float(currents[i]))
            spikes[i] = min(max(int(raw), 0), 1)
            s._voltages[i] = getattr(neuron, "v", 0.0)
    else
        for i, neuron in enumerate(s.neurons)
            raw = neuron.step(float(currents[i]))
            spikes[i] = min(max(int(raw), 0), 1)
            s._voltages[i] = getattr(neuron, "v", 0.0)
    return spikes
end

function reset_all(s::PopulationState)
    for neuron in s.neurons
        if hasattr(neuron, "reset")
            neuron.reset()
        elseif hasattr(neuron, "reset_state")
            neuron.reset_state()
    s._sync_voltages()
end

function get_states(s::PopulationState)
    if s.n == 0
        return {}
    sample = s.neurons[0]
    if hasattr(sample, "get_state")
        keys = sample.get_state().keys()
    elseif hasattr(sample, "__dataclass_fields__")
        keys = [k for k in sample.__dataclass_fields__ if k ! in ("dt",)]
    else
        keys = ["v"]
    result = {}
    for k in keys
        result[k] = collect([getattr(n, k, 0.0) for n in s.neurons])
    return result
end

function set_voltages(s::PopulationState, voltages)
    for i, neuron in enumerate(s.neurons)
        if hasattr(neuron, "v")
            neuron.v = float(voltages[i])
    s._voltages[:] = voltages[: s.n]
end

function voltages(s::PopulationState)
    return s._voltages
end

end # module PopulationAccel
