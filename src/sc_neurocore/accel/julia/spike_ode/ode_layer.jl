# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for spike_ode/ode_layer

module OdeLayerAccel

using Statistics, LinearAlgebra

mutable struct SpikingODELayerState
    tau_mem::Float64
    v_rest::Float64
    v_threshold::Float64
    v_reset::Float64
    C_mem::Float64
    n_inputs::Float64
    n_neurons::Float64
    dynamics::Float64
    dt_init::Float64
    dt_min::Float64
    max_steps::Float64
    W::Float64
    _v::Float64
end

function SpikingODELayerState()
    SpikingODELayerState(20.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function dvdt(s::SpikingODELayerState, v, I)
    return -(v - s.v_rest) / s.tau_mem + I / s.C_mem
end

function step(s::SpikingODELayerState, x, interval)
    I = s.W @ x
    spike_counts = zeros(s.n_neurons)
    t = 0.0
    dt = s.dt_init
    steps = 0
    while t < interval && steps < s.max_steps
        dt = min(dt, interval - t)
        if dt < s.dt_min
            break
        # Euler step
        dv = s.dynamics.dvdt(s._v, I)
        v_new = s._v + dt * dv
        # Event detection: threshold crossing
        crossed = v_new >= s.dynamics.v_threshold
        if crossed.any()
            # Bisection to find exact crossing time
            for _ in 1:5:  # 5 bisection steps
                dt_half = dt / 2
                v_mid = s._v + dt_half * dv
                still_crossed = v_mid >= s.dynamics.v_threshold
                if still_crossed.any()
                    dt = dt_half
                    v_new = v_mid
                else
                    break
            spike_counts[crossed] += 1
            v_new[crossed] = s.dynamics.v_reset
        s._v = v_new  # type: ignore[assignment]
        t += dt
        steps += 1
        # Adaptive step: increase if no spikes, decrease near threshold
        distance_to_thresh = s.dynamics.v_threshold - s._v
        min_dist = distance_to_thresh.min()
        if min_dist < 0.1 * s.dynamics.v_threshold
            dt = max(dt * 0.5, s.dt_min)
        else
            dt = min(dt * 1.5, s.dt_init)
    return spike_counts
end

function forward(s::SpikingODELayerState, inputs, interval)
    s.reset()
    T = inputs.shape[0]
    outputs = zeros((T, s.n_neurons))
    for t in 1:T
        outputs[t] = s.step(inputs[t], interval)
    return outputs
end

function reset(s::SpikingODELayerState)
    s._v = np.full(s.n_neurons, s.dynamics.v_rest)
end

function voltage(s::SpikingODELayerState)
    return s._v.copy()
end

end # module OdeLayerAccel
