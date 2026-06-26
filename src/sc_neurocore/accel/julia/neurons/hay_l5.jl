# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia candidate-first RK4 for hay_l5

module HayL5Accel

export step!, simulate, HayL5PyramidalNeuronState

const N_SUBSTEPS = 4

mutable struct HayL5PyramidalNeuronState
    v_s::Float64
    h_na::Float64
    n_k::Float64
    v_t::Float64
    m_ca::Float64
    h_ca::Float64
    m_ih::Float64
    v_a::Float64
    ca_a::Float64
    g_na::Float64
    g_k::Float64
    g_l_s::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    g_ca_t::Float64
    g_ih::Float64
    g_l_t::Float64
    e_ca::Float64
    e_ih::Float64
    g_ca_a::Float64
    g_kca::Float64
    g_l_a::Float64
    g_st::Float64
    g_ta::Float64
    p_s::Float64
    p_t::Float64
    p_a::Float64
    ca_decay::Float64
    f_ca::Float64
    dt::Float64
    v_threshold::Float64
    c_m::Float64
end

function HayL5PyramidalNeuronState()
    HayL5PyramidalNeuronState(-75.0, 0.9, 0.1, -75.0, 0.0, 1.0, 0.0, -75.0, 0.0001, 300.0, 40.0, 0.03, 50.0, -85.0, -75.0, 2.0, 0.02, 0.03, 140.0, -45.0, 1.5, 2.5, 0.03, 1.5, 0.8, 0.15, 0.25, 0.6, 200.0, 0.0002, 0.025, -30.0, 1.0)
end

@inline function _finite(values...)
    return all(isfinite, values)
end

@inline function _valid(s::HayL5PyramidalNeuronState)
    return _finite(s.v_s, s.h_na, s.n_k, s.v_t, s.m_ca, s.h_ca, s.m_ih, s.v_a, s.ca_a,
        s.g_na, s.g_k, s.g_l_s, s.e_na, s.e_k, s.e_l, s.g_ca_t, s.g_ih, s.g_l_t,
        s.e_ca, s.e_ih, s.g_ca_a, s.g_kca, s.g_l_a, s.g_st, s.g_ta, s.p_s, s.p_t,
        s.p_a, s.ca_decay, s.f_ca, s.dt, s.v_threshold, s.c_m) &&
        s.ca_a >= 0.0 && s.g_na >= 0.0 && s.g_k >= 0.0 && s.g_l_s >= 0.0 &&
        s.g_ca_t >= 0.0 && s.g_ih >= 0.0 && s.g_l_t >= 0.0 && s.g_ca_a >= 0.0 &&
        s.g_kca >= 0.0 && s.g_l_a >= 0.0 && s.g_st >= 0.0 && s.g_ta >= 0.0 &&
        s.f_ca >= 0.0 && s.p_s > 0.0 && s.p_t > 0.0 && s.p_a > 0.0 &&
        s.ca_decay > 0.0 && s.dt > 0.0 && s.c_m > 0.0
end

@inline function _derivatives(
    s::HayL5PyramidalNeuronState,
    state::NTuple{9,Float64},
    current_soma::Float64,
    current_tuft::Float64,
)
    v_s, h_na, n_k, v_t, m_ca, h_ca, m_ih, v_a, ca_a_raw = state
    ca_a = max(ca_a_raw, 0.0)
    m_na_inf = 1.0 / (1.0 + exp(-(v_s + 38.0) / 7.0))
    h_na_inf = 1.0 / (1.0 + exp((v_s + 65.0) / 6.0))
    n_k_inf = 1.0 / (1.0 + exp(-(v_s + 25.0) / 12.0))
    tau_h = 0.5 + 14.0 / (1.0 + exp((v_s + 35.0) / 10.0))
    tau_n = 1.0 + 5.0 / (1.0 + exp((v_s + 30.0) / 10.0))
    i_na = s.g_na * m_na_inf * m_na_inf * m_na_inf * h_na * (v_s - s.e_na)
    i_k = s.g_k * n_k * n_k * n_k * n_k * (v_s - s.e_k)
    i_l_s = s.g_l_s * (v_s - s.e_l)
    i_st = s.g_st * (v_s - v_t) / s.p_s

    m_ca_inf = 1.0 / (1.0 + exp(-(v_t + 27.0) / 7.0))
    h_ca_inf = 1.0 / (1.0 + exp((v_t + 52.0) / 5.0))
    m_ih_inf = 1.0 / (1.0 + exp((v_t + 75.0) / 5.5))
    i_ca_t = s.g_ca_t * m_ca * m_ca * h_ca * (v_t - s.e_ca)
    i_ih = s.g_ih * m_ih * (v_t - s.e_ih)
    i_l_t = s.g_l_t * (v_t - s.e_l)
    i_ts = s.g_st * (v_t - v_s) / s.p_t
    i_ta = s.g_ta * (v_t - v_a) / s.p_t

    m_ca_a_inf = 1.0 / (1.0 + exp(-(v_a + 30.0) / 5.0))
    kca_act = ca_a / (ca_a + 0.001)
    i_ca_a = s.g_ca_a * m_ca_a_inf * m_ca_a_inf * (v_a - s.e_ca)
    i_kca = s.g_kca * kca_act * (v_a - s.e_k)
    i_l_a = s.g_l_a * (v_a - s.e_l)
    i_at = s.g_ta * (v_a - v_t) / s.p_a

    return (
        (-i_na - i_k - i_l_s - i_st + current_soma / s.p_s) / s.c_m,
        (h_na_inf - h_na) / tau_h,
        (n_k_inf - n_k) / tau_n,
        (-i_ca_t - i_ih - i_l_t - i_ts - i_ta) / s.c_m,
        m_ca_inf - m_ca,
        (h_ca_inf - h_ca) / 20.0,
        (m_ih_inf - m_ih) / 50.0,
        (-i_ca_a - i_kca - i_l_a - i_at + current_tuft / s.p_a) / s.c_m,
        -s.f_ca * i_ca_a - ca_a / s.ca_decay,
    )
end

@inline function _rk4_substep(
    s::HayL5PyramidalNeuronState,
    state::NTuple{9,Float64},
    current_soma::Float64,
    current_tuft::Float64,
)
    dt = s.dt
    k1 = _derivatives(s, state, current_soma, current_tuft)
    s2 = ntuple(i -> state[i] + 0.5 * dt * k1[i], 9)
    k2 = _derivatives(s, s2, current_soma, current_tuft)
    s3 = ntuple(i -> state[i] + 0.5 * dt * k2[i], 9)
    k3 = _derivatives(s, s3, current_soma, current_tuft)
    s4 = ntuple(i -> state[i] + dt * k3[i], 9)
    k4 = _derivatives(s, s4, current_soma, current_tuft)
    next = ntuple(i -> state[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0, 9)
    return (next[1], next[2], next[3], next[4], next[5], next[6], next[7], next[8], max(next[9], 0.0))
end

function step!(s::HayL5PyramidalNeuronState, current_soma::Float64=0.0, current_tuft::Float64=0.0)
    if !_finite(current_soma, current_tuft) || !_valid(s)
        return 0
    end
    v_prev = s.v_s
    state = (s.v_s, s.h_na, s.n_k, s.v_t, s.m_ca, s.h_ca, s.m_ih, s.v_a, s.ca_a)
    for _ in 1:N_SUBSTEPS
        state = _rk4_substep(s, state, current_soma, current_tuft)
        if !_finite(state...)
            return 0
        end
    end
    s.v_s, s.h_na, s.n_k, s.v_t, s.m_ca, s.h_ca, s.m_ih, s.v_a, s.ca_a = state
    return (s.v_s >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; current_soma::Float64=10.0, current_tuft::Float64=0.0)
    s = HayL5PyramidalNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, current_soma, current_tuft)
        trace[t] = s.v_s
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module HayL5Accel
