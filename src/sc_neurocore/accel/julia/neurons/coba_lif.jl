# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for coba_lif

module CobaLifAccel

export step!, simulate, COBALIFNeuronState

mutable struct COBALIFNeuronState
    v::Float64
    g_e::Float64
    g_i::Float64
    c_m::Float64
    g_l::Float64
    e_l::Float64
    e_e::Float64
    e_i::Float64
    tau_e::Float64
    tau_i::Float64
    v_threshold::Float64
    v_reset::Float64
    dt::Float64
end

function COBALIFNeuronState()
    COBALIFNeuronState(-65.0, 0.0, 0.0, 200.0, 10.0, -65.0, 0.0, -80.0, 5.0, 10.0, -50.0, -65.0, 0.1)
end

function step!(s::COBALIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s.g_e += delta_ge
        s.g_i += delta_gi
        i_syn = s.g_e * (s.v - s.e_e) + s.g_i * (s.v - s.e_i)
        dv = (-s.g_l * (s.v - s.e_l) - i_syn + I_ext) / s.c_m * s.dt
        s.v += dv
        s.g_e *= exp(-s.dt / s.tau_e)
        s.g_i *= exp(-s.dt / s.tau_i)
        if s.v >= s.v_threshold
            s.v = s.v_reset
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = COBALIFNeuronState()
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

end # module CobaLifAccel
