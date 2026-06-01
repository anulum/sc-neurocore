# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for av_ron_cardiac

module AvRonCardiacAccel

export step!, simulate, reset!, AvRonCardiacNeuronState

mutable struct AvRonCardiacNeuronState
    v::Float64
    h::Float64
    n::Float64
    s::Float64
    g_na::Float64
    g_k::Float64
    g_s::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_s::Float64
    e_l::Float64
    dt::Float64
    v_threshold::Float64
end

function AvRonCardiacNeuronState()
    AvRonCardiacNeuronState(-60.0, 0.6, 0.3, 0.5, 80.0, 40.0, 20.0, 0.1, 40.0, -80.0, -25.0, -60.0, 0.02, -20.0)
end

finite_values(values::Float64...) = all(isfinite, values)
gate_in_range(value::Float64) = 0.0 <= value <= 1.0
bounded_exp(value::Float64) = exp(clamp(value, -745.0, 709.0))
sigmoid_pos(value::Float64) = 1.0 / (1.0 + bounded_exp(-value))
sigmoid_neg(value::Float64) = 1.0 / (1.0 + bounded_exp(value))

function valid_runtime(state::AvRonCardiacNeuronState)
    finite_values(state.v, state.h, state.n, state.s, state.g_na, state.g_k, state.g_s, state.g_l, state.e_na, state.e_k, state.e_s, state.e_l, state.dt, state.v_threshold) &&
        state.dt > 0.0 && state.g_na >= 0.0 && state.g_k >= 0.0 && state.g_s >= 0.0 && state.g_l >= 0.0 &&
        gate_in_range(state.h) && gate_in_range(state.n) && gate_in_range(state.s)
end

function rates(voltage::Float64)
    (
        sigmoid_pos((voltage + 40.0) / 7.0),
        sigmoid_neg((voltage + 45.0) / 5.0),
        sigmoid_pos((voltage + 40.0) / 15.0),
        sigmoid_neg((voltage + 35.0) / 3.0),
        1.0 + 12.0 * sigmoid_neg((voltage + 50.0) / 8.0),
        1.0 + 8.0 * sigmoid_neg((voltage + 35.0) / 8.0),
        200.0 + 1000.0 * sigmoid_neg((voltage + 30.0) / 5.0),
    )
end

function derivatives(state::AvRonCardiacNeuronState, candidate::NTuple{4, Float64}, I_ext::Float64)
    voltage, h_gate, n_gate, s_gate = candidate
    if !finite_values(voltage, h_gate, n_gate, s_gate) || !gate_in_range(h_gate) || !gate_in_range(n_gate) || !gate_in_range(s_gate)
        return (NaN, NaN, NaN, NaN)
    end
    m_inf, h_inf, n_inf, s_inf, tau_h, tau_n, tau_s = rates(voltage)
    i_na = state.g_na * m_inf^3 * h_gate * (voltage - state.e_na)
    i_k = state.g_k * n_gate^4 * (voltage - state.e_k)
    i_s = state.g_s * s_gate * (voltage - state.e_s)
    i_l = state.g_l * (voltage - state.e_l)
    (-i_na - i_k - i_s - i_l + I_ext, (h_inf - h_gate) / tau_h, (n_inf - n_gate) / tau_n, (s_inf - s_gate) / tau_s)
end

function add_scaled(state::NTuple{4, Float64}, slope::NTuple{4, Float64}, scale::Float64)
    (state[1] + scale * slope[1], state[2] + scale * slope[2], state[3] + scale * slope[3], state[4] + scale * slope[4])
end

function rk4_candidate(state::AvRonCardiacNeuronState, I_ext::Float64)
    old = (state.v, state.h, state.n, state.s)
    half_dt = 0.5 * state.dt
    k1 = derivatives(state, old, I_ext)
    k2 = derivatives(state, add_scaled(old, k1, half_dt), I_ext)
    k3 = derivatives(state, add_scaled(old, k2, half_dt), I_ext)
    k4 = derivatives(state, add_scaled(old, k3, state.dt), I_ext)
    candidate = (
        old[1] + state.dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        old[2] + state.dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0,
        old[3] + state.dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0,
        old[4] + state.dt * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]) / 6.0,
    )
    ok = finite_values(candidate...) && gate_in_range(candidate[2]) && gate_in_range(candidate[3]) && gate_in_range(candidate[4])
    return candidate, ok
end

function step!(state::AvRonCardiacNeuronState, I_ext::Float64=0.0; dt::Float64=state.dt)
    state.dt = dt
    if !isfinite(I_ext) || !valid_runtime(state)
        return 0
    end
    v_prev = state.v
    candidate, ok = rk4_candidate(state, I_ext)
    if !ok
        return 0
    end
    state.v, state.h, state.n, state.s = candidate
    return (state.v >= state.v_threshold && v_prev < state.v_threshold) ? 1 : 0
end

function reset!(state::AvRonCardiacNeuronState)
    state.v = -60.0
    state.h = 0.6
    state.n = 0.3
    state.s = 0.5
    return nothing
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.02)
    state = AvRonCardiacNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for step in 1:n_steps
        result = step!(state, I_ext; dt=dt)
        trace[step] = state.v
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module AvRonCardiacAccel
