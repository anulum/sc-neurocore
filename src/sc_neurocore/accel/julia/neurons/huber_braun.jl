# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for huber_braun

module HuberBraunAccel

export step!, simulate, HuberBraunNeuronState

mutable struct HuberBraunNeuronState
    v::Float64
    a_sd::Float64
    a_sr::Float64
    g_sd::Float64
    g_sr::Float64
    g_l::Float64
    e_sd::Float64
    e_sr::Float64
    e_l::Float64
    tau_sd::Float64
    tau_sr::Float64
    eta::Float64
    dt::Float64
    v_threshold::Float64
end

function HuberBraunNeuronState()
    HuberBraunNeuronState(-50.0, 0.0, 0.0, 1.5, 0.4, 0.1, 50.0, -90.0, -60.0, 10.0, 20.0, 0.012, 0.1, -20.0)
end

function step!(s::HuberBraunNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        sd_inf = 1.0 / (1.0 + exp(-(s.v + 40.0) / 6.0))
        sr_inf = 1.0 / (1.0 + exp((s.v + 40.0) / 6.0))
        s.a_sd += (sd_inf - s.a_sd) / s.tau_sd * s.dt
        s.a_sr += (sr_inf - s.a_sr) / s.tau_sr * s.dt
        i_sd = s.g_sd * s.a_sd * (s.v - s.e_sd)
        i_sr = s.g_sr * s.a_sr * (s.v - s.e_sr)
        i_l = s.g_l * (s.v - s.e_l)
        s.v += (-i_sd - i_sr - i_l + I_ext + s.eta * randn()) * s.dt
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = HuberBraunNeuronState()
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

end # module HuberBraunAccel
