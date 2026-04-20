# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for alpha_motor_neuron

module AlphaMotorNeuronAccel

export step!, simulate, AlphaMotorNeuronState

mutable struct AlphaMotorNeuronState
    v::Float64
    h::Float64
    n::Float64
    m_pic::Float64
    h_pic::Float64
    ca::Float64
    ca_buf::Float64
    g_na::Float64
    g_k::Float64
    g_pic::Float64
    g_ahp::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_ca::Float64
    e_l::Float64
    c_m::Float64
    phi::Float64
    tau_ca::Float64
    buf_ratio::Float64
    dt::Float64
    v_threshold::Float64
end

function AlphaMotorNeuronState()
    AlphaMotorNeuronState(-65.0, 0.8, 0.1, 0.0, 1.0, 0.0, 0.0, 35.0, 9.0, 0.15, 3.0, 0.3, 55.0, -90.0, 120.0, -65.0, 1.5, 4.0, 150.0, 0.003, 0.01, -20.0)
end

function step!(s::AlphaMotorNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    v_prev = s.v
    n_sub = max(1, Int(0.5 / max(s.dt, 0.001)))
    for _ in 1:n_sub
        am = _safe_rate(0.1, 35.0, s.v, 10.0, 1.0)
        bm = 4.0 * exp(-(s.v + 60.0) / 18.0)
        m_inf = am / (am + bm)
        ah = 0.07 * exp(-(s.v + 58.0) / 20.0)
        bh = 1.0 / (1.0 + exp(-(s.v + 28.0) / 10.0))
        an = _safe_rate(0.01, 34.0, s.v, 10.0, 0.1)
        bn = 0.125 * exp(-(s.v + 44.0) / 80.0)
        s.h += s.phi * (ah * (1.0 - s.h) - bh * s.h) * s.dt
        s.n += s.phi * (an * (1.0 - s.n) - bn * s.n) * s.dt
        m_pic_inf = 1.0 / (1.0 + exp(-(s.v + 40.0) / 5.0))
        s.m_pic += (m_pic_inf - s.m_pic) / 50.0 * s.dt
        h_pic_inf = 1.0 / (1.0 + exp((s.v + 40.0) / 8.0))
        tau_h_pic = 200.0 + 100.0 / max(0.01, 1.0 + ((s.v + 40.0) / 10.0) ^ 2)
        s.h_pic += (h_pic_inf - s.h_pic) / tau_h_pic * s.dt
        s.h_pic = max(0.0, min(1.0, s.h_pic))
        i_ca_entry = s.g_pic * s.m_pic * s.h_pic * (s.v - s.e_ca)
        ca_influx = (i_ca_entry < 0.0) ? -i_ca_entry * 0.001 : 0.0
        ca_spike = (s.v > -10.0) ? 0.02 : 0.0
        free_ca_change = (ca_influx + ca_spike) * s.buf_ratio
        s.ca += (-s.ca / s.tau_ca + free_ca_change) * s.dt
        s.ca = max(0.0, s.ca)
        s.ca_buf += ((ca_influx + ca_spike) * (1.0 - s.buf_ratio) - s.ca_buf / (s.tau_ca * 5.0)) * s.dt
        s.ca_buf = max(0.0, s.ca_buf)
        ca_total = s.ca + s.ca_buf * 0.01
        ahp_inf = ca_total ^ 2 / (ca_total ^ 2 + 0.25)
        i_na = s.g_na * m_inf ^ 3 * s.h * (s.v - s.e_na)
        i_k = s.g_k * s.n ^ 4 * (s.v - s.e_k)
        i_pic = s.g_pic * s.m_pic * s.h_pic * (s.v - s.e_ca)
        i_ahp = s.g_ahp * ahp_inf * (s.v - s.e_k)
        i_l = s.g_l * (s.v - s.e_l)
        s.v += (-i_na - i_k - i_pic - i_ahp - i_l + I_ext) / s.c_m * s.dt
    end
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = AlphaMotorNeuronState()
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

end # module AlphaMotorNeuronAccel
