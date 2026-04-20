# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for tsodyks_markram

module TsodyksMarkramAccel

export step!, simulate, TsodyksMarkramNeuronState

mutable struct TsodyksMarkramNeuronState
    v::Float64
    x::Float64
    u::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    tau_m::Float64
    tau_d::Float64
    tau_f::Float64
    u_se::Float64
    a_se::Float64
    r_m::Float64
    dt::Float64
end

function TsodyksMarkramNeuronState()
    TsodyksMarkramNeuronState(-65.0, 1.0, 0.2, -65.0, -65.0, -50.0, 20.0, 200.0, 600.0, 0.2, 50.0, 1.0, 0.1)
end

function step!(s::TsodyksMarkramNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s.x += (1.0 - s.x) / s.tau_d * s.dt
        s.u += (s.u_se - s.u) / s.tau_f * s.dt
        i_syn = 0.0
        if presynaptic_spike
            s.u += s.u_se * (1.0 - s.u)
            i_syn = s.a_se * s.u * s.x
            s.x -= s.u * s.x
        end
        dv = (-(s.v - s.v_rest) + s.r_m * (i_syn + I_ext)) / s.tau_m * s.dt
        s.v += dv
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
    s = TsodyksMarkramNeuronState()
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

end # module TsodyksMarkramAccel
