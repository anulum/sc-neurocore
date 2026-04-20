# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for brunel_wang

module BrunelWangAccel

export step!, simulate, BrunelWangNeuronState

mutable struct BrunelWangNeuronState
    v::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    tau_m::Float64
    tau_ref::Float64
    tau_ampa::Float64
    tau_nmda_rise::Float64
    tau_nmda_decay::Float64
    tau_gaba::Float64
    g_ampa_ext::Float64
    g_ampa_rec::Float64
    g_nmda::Float64
    g_gaba::Float64
    v_ampa::Float64
    v_nmda::Float64
    v_gaba::Float64
    C_m::Float64
    mg_conc::Float64
    dt::Float64
end

function BrunelWangNeuronState()
    BrunelWangNeuronState(-70.0, -70.0, -55.0, -50.0, 20.0, 2.0, 2.0, 2.0, 100.0, 5.0, 2.1, 0.05, 0.165, 1.3, 0.0, 0.0, -70.0, 0.5, 1.0, 0.1)
end

function _nmda_voltage_dep(s::BrunelWangNeuronState, v)
    return 1.0 / (1.0 + s.mg_conc / 3.57 * exp(-0.062 * v))
end

function get_state(s::BrunelWangNeuronState)
    return {'v": s.v, "ref_remaining': s._ref_remaining}
end

function step!(s::BrunelWangNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        if s._ref_remaining > 0
            s._ref_remaining -= s.dt
            return 0
        end
        i_ampa = -s.g_ampa_ext * (s.v - s.v_ampa) * i_ampa_ext
        i_ampa += -s.g_ampa_rec * (s.v - s.v_ampa) * s_ampa_rec
        i_nmda = -s.g_nmda * s._nmda_voltage_dep(s.v) * (s.v - s.v_nmda) * s_nmda_rec
        i_gaba = -s.g_gaba * (s.v - s.v_gaba) * s_gaba
        i_leak = -(s.v - s.v_rest) / s.tau_m
        dv = (i_leak + (i_ampa + i_nmda + i_gaba) / s.C_m) * s.dt
        s.v += dv
        if s.v >= s.v_threshold
            s.v = s.v_reset
            s._ref_remaining = s.tau_ref
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = BrunelWangNeuronState()
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

end # module BrunelWangAccel
