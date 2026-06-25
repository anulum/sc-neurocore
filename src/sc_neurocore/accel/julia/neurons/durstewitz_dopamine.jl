# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia candidate-first RK4 mirror for durstewitz_dopamine

module DurstewitzDopamineAccel

export step!, simulate, DurstewitzDopamineNeuronState

mutable struct DurstewitzDopamineNeuronState
    v::Float64
    h_na::Float64
    n_k::Float64
    g_na::Float64
    g_k::Float64
    g_nmda::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_nmda::Float64
    e_l::Float64
    mg::Float64
    d1_level::Float64
    g_nmda_scale::Float64
    g_k_scale::Float64
    v_shift_na::Float64
    dt::Float64
    v_threshold::Float64
end

function DurstewitzDopamineNeuronState()
    DurstewitzDopamineNeuronState(-65.0, 0.7, 0.2, 45.0, 18.0, 0.5, 0.02, 55.0, -80.0, 0.0, -65.0, 1.0, 0.0, 2.5, 1.5, -5.0, 0.05, -20.0)
end

# Right-hand side (dV, dh_na, dn_k) at one consistent state. The sodium
# activation m_inf is instantaneous; the conductance powers use explicit
# multiplication and the Mg2+ block keeps the mg/3.57*exp operand order so the
# Python, Rust, Go, and Mojo backends reproduce the trajectory bit-for-bit.
function _derivatives(s::DurstewitzDopamineNeuronState, v::Float64, h_na::Float64, n_k::Float64, I_ext::Float64)
    v_sh = s.d1_level * s.v_shift_na
    m_na_inf = 1.0 / (1.0 + exp(-(v + 30.0 + v_sh) / 9.5))
    h_na_inf = 1.0 / (1.0 + exp((v + 53.0) / 7.0))
    n_k_inf = 1.0 / (1.0 + exp(-(v + 30.0) / 10.0))
    tau_h = 0.5 + 14.0 / (1.0 + exp((v + 50.0) / 12.0))
    tau_n = 1.0 + 11.0 / (1.0 + exp((v + 40.0) / 10.0))
    d_h_na = (h_na_inf - h_na) / tau_h
    d_n_k = (n_k_inf - n_k) / tau_n
    mg_block = 1.0 / (1.0 + s.mg / 3.57 * exp(-0.062 * v))
    nmda_g = s.g_nmda * (1.0 + s.d1_level * (s.g_nmda_scale - 1.0))
    k_g = s.g_k * (1.0 + s.d1_level * (s.g_k_scale - 1.0))
    i_na = s.g_na * m_na_inf * m_na_inf * m_na_inf * h_na * (v - s.e_na)
    i_k = k_g * n_k * n_k * n_k * n_k * (v - s.e_k)
    i_nmda = nmda_g * mg_block * (v - s.e_nmda)
    i_l = s.g_l * (v - s.e_l)
    d_v = -i_na - i_k - i_nmda - i_l + I_ext
    return d_v, d_h_na, d_n_k
end

function _rk4_substep(s::DurstewitzDopamineNeuronState, v::Float64, h_na::Float64, n_k::Float64, I_ext::Float64)
    dt = s.dt
    k1v, k1h, k1n = _derivatives(s, v, h_na, n_k, I_ext)
    k2v, k2h, k2n = _derivatives(s, v + 0.5 * dt * k1v, h_na + 0.5 * dt * k1h, n_k + 0.5 * dt * k1n, I_ext)
    k3v, k3h, k3n = _derivatives(s, v + 0.5 * dt * k2v, h_na + 0.5 * dt * k2h, n_k + 0.5 * dt * k2n, I_ext)
    k4v, k4h, k4n = _derivatives(s, v + dt * k3v, h_na + dt * k3h, n_k + dt * k3n, I_ext)
    next_v = v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
    next_h = h_na + dt * (k1h + 2.0 * k2h + 2.0 * k3h + k4h) / 6.0
    next_n = n_k + dt * (k1n + 2.0 * k2n + 2.0 * k3n + k4n) / 6.0
    return next_v, next_h, next_n
end

function step!(s::DurstewitzDopamineNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)
    v_prev = s.v
    next_v, next_h, next_n = _rk4_substep(s, s.v, s.h_na, s.n_k, I_ext)
    s.v = next_v
    s.h_na = next_h
    s.n_k = next_n
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.05)
    s = DurstewitzDopamineNeuronState()
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

end # module DurstewitzDopamineAccel
