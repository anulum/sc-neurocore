# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for dendrify

module DendrifyAccel

export step!, simulate, DendrifyNeuronState

mutable struct DendrifyNeuronState
    v_s::Float64
    v_d::Float64
    d_active::Float64
    tau_s::Float64
    tau_d::Float64
    g_c::Float64
    d_threshold::Float64
    d_amplitude::Float64
    d_duration::Float64
    d_timer::Float64
    v_rest::Float64
    v_threshold::Float64
    v_reset::Float64
    dt::Float64
end

function DendrifyNeuronState()
    DendrifyNeuronState(-65.0, -65.0, 0.0, 10.0, 20.0, 0.8, -35.0, 30.0, 10.0, 0.0, -65.0, -50.0, -65.0, 0.1)
end

function step!(s::DendrifyNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_s_prev = s.v_s
        dv_d = (-(s.v_d - s.v_rest) + I_ext - s.g_c * (s.v_d - s.v_s)) / s.tau_d
        s.v_d += dv_d * s.dt
        if ! s.d_active && s.v_d >= s.d_threshold
            s.d_active = true
            s.d_timer = s.d_duration
        end
        if s.d_active
            s.d_timer -= s.dt
            d_inject = s.d_amplitude
            if s.d_timer <= 0.0
                s.d_active = false
            end
        else
            d_inject = 0.0
        end
        dv_s = (-(s.v_s - s.v_rest) + s.g_c * (s.v_d - s.v_s) + d_inject) / s.tau_s
        s.v_s += dv_s * s.dt
        if s.v_s >= s.v_threshold && v_s_prev < s.v_threshold
            s.v_s = s.v_reset
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = DendrifyNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v_s
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module DendrifyAccel
