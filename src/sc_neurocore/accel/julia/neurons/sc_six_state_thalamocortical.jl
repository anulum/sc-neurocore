# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — preserved Julia six-state thalamocortical recurrence

module SCSixStateThalamocorticalAccel

export step!, simulate, SCSixStateThalamocorticalNeuronState

mutable struct SCSixStateThalamocorticalNeuronState
    v::Float64
    h_na::Float64
    n_k::Float64
    m_h::Float64
    h_t::Float64
    na_i::Float64
    g_na::Float64
    g_k::Float64
    g_h::Float64
    g_t::Float64
    g_kna::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_h::Float64
    e_ca::Float64
    e_l::Float64
    na_pump_max::Float64
    na_eq::Float64
    dt::Float64
    v_threshold::Float64
end

function SCSixStateThalamocorticalNeuronState()
    SCSixStateThalamocorticalNeuronState(-65.0, 0.6, 0.3, 0.0, 0.9, 5.0, 50.0, 5.0, 1.0, 3.0, 1.33, 0.02, 50.0, -90.0, -43.0, 120.0, -70.0, 20.0, 9.5, 0.05, -20.0)
end

# Right-hand side (dV, dh_na, dn_k, dm_h, dh_t, dna_i) at one consistent state.
# The conductance powers use explicit multiplication and the I_KNa Hill exponent
# 3.5 is evaluated as b*b*b*sqrt(b) so Python/Rust/Go/Mojo reproduce it bit-for-bit.
function _derivatives(s::SCSixStateThalamocorticalNeuronState, v::Float64, h_na::Float64, n_k::Float64, m_h::Float64, h_t::Float64, na_i::Float64, I_ext::Float64)
    m_na_inf = 1.0 / (1.0 + exp(-(v + 38.0) / 10.0))
    h_na_inf = 1.0 / (1.0 + exp((v + 43.0) / 6.0))
    n_k_inf = 1.0 / (1.0 + exp(-(v + 27.0) / 11.5))
    m_h_inf = 1.0 / (1.0 + exp((v + 75.0) / 5.5))
    m_t_inf = 1.0 / (1.0 + exp(-(v + 59.0) / 6.2))
    h_t_inf = 1.0 / (1.0 + exp((v + 83.0) / 4.0))
    hill_base = 38.7 / max(na_i, 0.01)
    w_kna = 0.37 / (1.0 + hill_base * hill_base * hill_base * sqrt(hill_base))
    tau_h_na = max(1.0 + 10.0 / (1.0 + exp((v + 40.0) / 10.0)), 0.1)
    z_n = (v + 50.0) / 25.0
    tau_n_k = max(5.0 + 47.0 * exp(-(z_n * z_n)), 0.1)
    tau_m_h = max(20.0 + 1000.0 / (exp((v + 71.5) / 14.2) + exp(-(v + 89.0) / 11.6)), 1.0)
    tau_h_t = (v < -81.0) ? max(30.8 + 211.4 * exp((v + 115.2) / 5.0) / (1.0 + exp((v + 86.0) / 3.2)), 0.1) : 10.0
    d_h_na = (h_na_inf - h_na) / tau_h_na
    d_n_k = (n_k_inf - n_k) / tau_n_k
    d_m_h = (m_h_inf - m_h) / tau_m_h
    d_h_t = (h_t_inf - h_t) / tau_h_t
    i_na = s.g_na * m_na_inf * m_na_inf * m_na_inf * h_na * (v - s.e_na)
    i_k = s.g_k * n_k * n_k * n_k * n_k * (v - s.e_k)
    i_h = s.g_h * m_h * (v - s.e_h)
    i_t = s.g_t * m_t_inf * m_t_inf * h_t * (v - s.e_ca)
    i_kna = s.g_kna * w_kna * (v - s.e_k)
    i_l = s.g_l * (v - s.e_l)
    d_v = -i_na - i_k - i_h - i_t - i_kna - i_l + I_ext
    d_na_i = -0.001 * i_na - s.na_pump_max * (na_i / (na_i + s.na_eq))
    return d_v, d_h_na, d_n_k, d_m_h, d_h_t, d_na_i
end

function _rk4_substep(s::SCSixStateThalamocorticalNeuronState, y::NTuple{6,Float64}, I_ext::Float64)
    dt = s.dt
    k1 = _derivatives(s, y[1], y[2], y[3], y[4], y[5], y[6], I_ext)
    k2 = _derivatives(s, y[1] + 0.5 * dt * k1[1], y[2] + 0.5 * dt * k1[2], y[3] + 0.5 * dt * k1[3], y[4] + 0.5 * dt * k1[4], y[5] + 0.5 * dt * k1[5], y[6] + 0.5 * dt * k1[6], I_ext)
    k3 = _derivatives(s, y[1] + 0.5 * dt * k2[1], y[2] + 0.5 * dt * k2[2], y[3] + 0.5 * dt * k2[3], y[4] + 0.5 * dt * k2[4], y[5] + 0.5 * dt * k2[5], y[6] + 0.5 * dt * k2[6], I_ext)
    k4 = _derivatives(s, y[1] + dt * k3[1], y[2] + dt * k3[2], y[3] + dt * k3[3], y[4] + dt * k3[4], y[5] + dt * k3[5], y[6] + dt * k3[6], I_ext)
    return ntuple(i -> y[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0, 6)
end

function step!(s::SCSixStateThalamocorticalNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)
    v_prev = s.v
    y = (s.v, s.h_na, s.n_k, s.m_h, s.h_t, s.na_i)
    y = _rk4_substep(s, y, I_ext)
    s.v = y[1]
    s.h_na = y[2]
    s.n_k = y[3]
    s.m_h = y[4]
    s.h_t = y[5]
    s.na_i = max(y[6], 0.0)
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.05)
    s = SCSixStateThalamocorticalNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext)
        trace[t] = s.v
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module SCSixStateThalamocorticalAccel
