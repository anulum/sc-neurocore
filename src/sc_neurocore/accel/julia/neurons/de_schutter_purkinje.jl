# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia candidate-first RK4 for de_schutter_purkinje

module DeSchutterPurkinjeAccel

export step!, simulate, DeSchutterPurkinjeNeuronState

const N_SUBSTEPS = 5

mutable struct DeSchutterPurkinjeNeuronState
    v::Float64
    h_na::Float64
    n_k::Float64
    m_cap::Float64
    h_cap::Float64
    q_kca::Float64
    ca::Float64
    g_na::Float64
    g_k::Float64
    g_cap::Float64
    g_kca::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_ca::Float64
    e_l::Float64
    ca_decay::Float64
    f_ca::Float64
    dt::Float64
    v_threshold::Float64
end

function DeSchutterPurkinjeNeuronState()
    DeSchutterPurkinjeNeuronState(-68.0, 0.8, 0.1, 0.0, 0.9, 0.0, 0.0001, 125.0, 10.0, 45.0, 35.0, 0.5, 45.0, -85.0, 135.0, -68.0, 0.02, 0.00024, 0.01, -20.0)
end

@inline function _finite(values...)
    return all(isfinite, values)
end

@inline function _valid(s::DeSchutterPurkinjeNeuronState)
    return _finite(s.v, s.h_na, s.n_k, s.m_cap, s.h_cap, s.q_kca, s.ca, s.g_na, s.g_k, s.g_cap, s.g_kca, s.g_l, s.e_na, s.e_k, s.e_ca, s.e_l, s.ca_decay, s.f_ca, s.dt, s.v_threshold) &&
        s.ca >= 0.0 && s.g_na >= 0.0 && s.g_k >= 0.0 && s.g_cap >= 0.0 &&
        s.g_kca >= 0.0 && s.g_l >= 0.0 && s.ca_decay >= 0.0 && s.f_ca >= 0.0 &&
        s.dt > 0.0
end

@inline function _derivatives(
    s::DeSchutterPurkinjeNeuronState,
    v::Float64,
    h_na::Float64,
    n_k::Float64,
    m_cap::Float64,
    h_cap::Float64,
    q_kca::Float64,
    ca::Float64,
    current::Float64,
)
    ca_eff = max(ca, 0.0)
    m_na_inf = 1.0 / (1.0 + exp(-(v + 35.0) / 7.5))
    h_na_inf = 1.0 / (1.0 + exp((v + 55.0) / 7.0))
    n_k_inf = 1.0 / (1.0 + exp(-(v + 30.0) / 15.0))
    m_cap_inf = 1.0 / (1.0 + exp(-(v + 19.0) / 5.5))
    h_cap_inf = 1.0 / (1.0 + exp((v + 48.0) / 7.0))
    q_kca_inf = ca_eff / (ca_eff + 0.0002)
    tau_h_na = 0.5 + 14.0 / (1.0 + exp((v + 40.0) / 12.0))
    tau_n_k = 1.0 + 11.0 / (1.0 + exp((v + 15.0) / 8.0))
    d_h_na = (h_na_inf - h_na) / tau_h_na
    d_n_k = (n_k_inf - n_k) / tau_n_k
    d_m_cap = (m_cap_inf - m_cap) / 0.3
    d_h_cap = (h_cap_inf - h_cap) / 45.0
    d_q_kca = q_kca_inf - q_kca
    i_na = s.g_na * m_na_inf * m_na_inf * m_na_inf * h_na * (v - s.e_na)
    i_k = s.g_k * n_k * n_k * n_k * n_k * (v - s.e_k)
    i_cap = s.g_cap * m_cap * m_cap * h_cap * (v - s.e_ca)
    i_kca = s.g_kca * q_kca * (v - s.e_k)
    i_l = s.g_l * (v - s.e_l)
    d_v = -i_na - i_k - i_cap - i_kca - i_l + current
    d_ca = -s.f_ca * i_cap - s.ca_decay * ca_eff
    return d_v, d_h_na, d_n_k, d_m_cap, d_h_cap, d_q_kca, d_ca
end

@inline function _rk4_substep(
    s::DeSchutterPurkinjeNeuronState,
    state::NTuple{7,Float64},
    current::Float64,
)
    dt = s.dt
    k1 = _derivatives(s, state..., current)
    s2 = ntuple(i -> state[i] + 0.5 * dt * k1[i], 7)
    k2 = _derivatives(s, s2..., current)
    s3 = ntuple(i -> state[i] + 0.5 * dt * k2[i], 7)
    k3 = _derivatives(s, s3..., current)
    s4 = ntuple(i -> state[i] + dt * k3[i], 7)
    k4 = _derivatives(s, s4..., current)
    next = ntuple(i -> state[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0, 7)
    return (next[1], next[2], next[3], next[4], next[5], next[6], max(next[7], 0.0))
end

function step!(s::DeSchutterPurkinjeNeuronState, I_ext::Float64=0.0; dt::Float64=0.01)
    if !_finite(I_ext) || !_valid(s)
        return 0
    end
    v_prev = s.v
    state = (s.v, s.h_na, s.n_k, s.m_cap, s.h_cap, s.q_kca, s.ca)
    for _ in 1:N_SUBSTEPS
        state = _rk4_substep(s, state, I_ext)
        if !_finite(state...)
            return 0
        end
    end
    s.v, s.h_na, s.n_k, s.m_cap, s.h_cap, s.q_kca, s.ca = state
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=500.0, dt::Float64=0.01)
    s = DeSchutterPurkinjeNeuronState()
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

end # module DeSchutterPurkinjeAccel
