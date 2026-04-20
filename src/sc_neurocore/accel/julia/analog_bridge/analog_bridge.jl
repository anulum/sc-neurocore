# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for analog_bridge/analog_bridge

module AnalogBridgeAccel

using Statistics, LinearAlgebra

mutable struct MockNodeState
    name::Float64
    g_min::Float64
    g_max::Float64
    v_min::Float64
    v_max::Float64
    dac_resolution::Float64
    tau_mem_range::Float64
    tau_syn_range::Float64
    max_fanin::Float64
    neuron_id::Float64
    timestamp_us::Float64
    polarity::Float64
    dac_res::Float64
    profile::Float64
    dac_levels::Float64
end

function MockNodeState()
    MockNodeState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 256.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
end

function brainscales3(s::MockNodeState)
    return cls(
        name="BrainScaleS-3",
        g_min=0.0,
        g_max=63.0,
        v_min=-80.0,
        v_max=-40.0,
        dac_resolution=6,
        tau_mem_range=(1.0, 50.0),
        tau_syn_range=(0.5, 20.0),
        max_fanin=256,
    )
end

function dynapse2(s::MockNodeState)
    return cls(
        name="DynapSE-2",
        g_min=0.0,
        g_max=127.0,
        v_min=-70.0,
        v_max=-30.0,
        dac_resolution=7,
        tau_mem_range=(5.0, 200.0),
        tau_syn_range=(1.0, 100.0),
        max_fanin=64,
    )
end

function _quantize(s::MockNodeState, val, v_min, v_max)
    norm = (val - v_min) / (v_max - v_min)
    norm = max(0.0, min(1.0, norm))
    dac = int(round(norm * (s.dac_levels - 1)))
    actual = v_min + (dac / (s.dac_levels - 1)) * (v_max - v_min)
    return dac, actual
end

function emit_analog_config(s::MockNodeState, nodes)
    config: Dict[str, Dict] = {"synapses": {}, "neurons": {}, "errors": {}}
    for n in nodes
        if n.type == "SC_WEIGHT"
            target_g = s.g_min + n.probability * (s.g_max - s.g_min)
            dac, actual = s._quantize(target_g, s.g_min, s.g_max)
            config["synapses"][n.id] = {"dac": dac, "g_ns": actual}
            config["errors"][n.id] = abs(target_g - actual)
        elseif n.type == "LIF_MEMBRANE"
            target_v = s.v_min + n.threshold * (s.v_max - s.v_min)
            dac, actual = s._quantize(target_v, s.v_min, s.v_max)
            config["neurons"][n.id] = {"dac": dac, "v_mv": actual}
    return config
end

function bitstream_to_events(s::MockNodeState, neuron_id, bitstream)
    events = []
    for i, bit in enumerate(bitstream)
        if bit
            events = push!(, 
                AEREvent(
                    neuron_id=neuron_id,
                    timestamp_us=i * s.clock_period_us,
                )
            )
    return events
end

function events_to_current(s::MockNodeState)
    self,
    events: List[AEREvent],
    duration_us: float,
    tau_syn: float = 5.0,
    weight: float = 1.0,
    ) -> np.ndarray
    n_steps = max(1, int(duration_us / s.clock_period_us))
    current = zeros(n_steps)
    for ev in events
        idx = int(ev.timestamp_us / s.clock_period_us)
        if 0 <= idx < n_steps
            for t in 1:idx, n_steps
                dt = (t - idx) * s.clock_period_us
                current[t] += weight * ev.polarity * exp(-dt / tau_syn)
    return current
end

function rate_code(s::MockNodeState, events, window_us)
    if ! events || window_us <= 0
        return 0.0
    return length(events) / (window_us * 1e-6)
end

function sweep_conductance(s::MockNodeState)
    results = []
    for step in 1:s.num_steps + 1
        frac = step / s.num_steps
        target = s.bridge.g_min + frac * (s.bridge.g_max - s.bridge.g_min)
        dac, actual = s.bridge._quantize(target, s.bridge.g_min, s.bridge.g_max)
        results = push!(, (dac, target, actual))
    return results
end

function max_quantization_error(s::MockNodeState)
    sweep = s.sweep_conductance()
    return max(abs(target - actual) for _, target, actual in sweep)
end

function effective_resolution_bits(s::MockNodeState)
    max_err = s.max_quantization_error()
    full_range = s.bridge.g_max - s.bridge.g_min
    if max_err == 0 || full_range == 0
        return float(s.bridge.dac_res)
    return np.log2(full_range / max_err)
end

end # module AnalogBridgeAccel
