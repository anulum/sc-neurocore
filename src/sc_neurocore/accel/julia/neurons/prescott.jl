# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia mirror for Prescott 2008 RK4 dynamics

module PrescottAccel

export step!, simulate, PrescottNeuronState

mutable struct PrescottNeuronState
    v::Float64
    w::Float64
    g_fast::Float64
    g_slow::Float64
    g_l::Float64
    e_fast::Float64
    e_slow::Float64
    e_l::Float64
    beta_w::Float64
    gamma_w::Float64
    tau_w::Float64
    phi::Float64
    dt::Float64
    v_threshold::Float64
end

function PrescottNeuronState()
    PrescottNeuronState(-65.0, 0.0, 20.0, 20.0, 2.0, 50.0, -100.0, -70.0, -21.0, 15.0, 100.0, 0.15, 0.1, -20.0)
end

sigmoid(x::Float64)::Float64 = x >= 0.0 ? begin
    z = exp(-x)
    1.0 / (1.0 + z)
end : begin
    z = exp(x)
    z / (1.0 + z)
end

valid_state(v::Float64, w::Float64)::Bool = isfinite(v) && isfinite(w) && 0.0 <= w <= 1.0

function valid_runtime(s::PrescottNeuronState)::Bool
    valid_state(s.v, s.w) &&
        isfinite(s.g_fast) && s.g_fast >= 0.0 &&
        isfinite(s.g_slow) && s.g_slow >= 0.0 &&
        isfinite(s.g_l) && s.g_l >= 0.0 &&
        isfinite(s.e_fast) && isfinite(s.e_slow) && isfinite(s.e_l) &&
        isfinite(s.beta_w) && isfinite(s.gamma_w) && s.gamma_w > 0.0 &&
        isfinite(s.tau_w) && s.tau_w > 0.0 &&
        isfinite(s.phi) && s.phi >= 0.0 &&
        isfinite(s.dt) && s.dt > 0.0 && isfinite(s.v_threshold)
end

function derivatives(s::PrescottNeuronState, v::Float64, w::Float64, i_ext::Float64)
    valid_state(v, w) || return (0.0, 0.0, false)
    m_inf = sigmoid((v + 20.0) / 15.0)
    w_inf = sigmoid((v - s.beta_w) / s.gamma_w)
    i_fast = s.g_fast * m_inf * (v - s.e_fast)
    i_slow = s.g_slow * w * (v - s.e_slow)
    i_l = s.g_l * (v - s.e_l)
    dv = -i_fast - i_slow - i_l + i_ext
    dw = s.phi * (w_inf - w) / s.tau_w
    return (dv, dw, isfinite(dv) && isfinite(dw))
end

function rk4_step(s::PrescottNeuronState, i_ext::Float64)
    dt = s.dt
    k1_v, k1_w, ok = derivatives(s, s.v, s.w, i_ext)
    ok || return (0.0, 0.0, false)
    k2_v, k2_w, ok = derivatives(s, s.v + 0.5 * dt * k1_v, s.w + 0.5 * dt * k1_w, i_ext)
    ok || return (0.0, 0.0, false)
    k3_v, k3_w, ok = derivatives(s, s.v + 0.5 * dt * k2_v, s.w + 0.5 * dt * k2_w, i_ext)
    ok || return (0.0, 0.0, false)
    k4_v, k4_w, ok = derivatives(s, s.v + dt * k3_v, s.w + dt * k3_w, i_ext)
    ok || return (0.0, 0.0, false)
    next_v = s.v + dt * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v) / 6.0
    next_w = s.w + dt * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w) / 6.0
    return (next_v, next_w, valid_state(next_v, next_w))
end

function step!(s::PrescottNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    isfinite(I_ext) || return -1
    valid_runtime(s) || return -1
    v_prev = s.v
    next_v, next_w, ok = rk4_step(s, I_ext)
    ok || return -1
    s.v = next_v
    s.w = next_w
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = PrescottNeuronState()
    s.dt = dt
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module PrescottAccel
