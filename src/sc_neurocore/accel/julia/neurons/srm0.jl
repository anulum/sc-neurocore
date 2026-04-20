# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for srm0

module Srm0Accel

export step!, simulate, SRM0NeuronState

mutable struct SRM0NeuronState
    v::Float64
    v_rest::Float64
    v_threshold::Float64
    tau_m::Float64
    tau_eta::Float64
    eta_reset::Float64
    resistance::Float64
    dt::Float64
end

function SRM0NeuronState()
    SRM0NeuronState(0.0, 0.0, 1.0, 20.0, 50.0, 5.0, 1.0, 1.0)
end

function get_state(s::SRM0NeuronState)
    return {'v": s.v, "eta": s._eta, "t': s._t}
end

function step!(s::SRM0NeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s._eta *= exp(-s.dt / s.tau_eta)
        effective_rest = s.v_rest + s._eta
        dv = (s.resistance * I_ext - (s.v - effective_rest)) * s.dt / s.tau_m
        s.v += dv
        s._t += s.dt
        if s.v >= s.v_threshold
            s.v = s.v_rest
            s._eta = -s.eta_reset
            s._last_spike_time = s._t
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = SRM0NeuronState()
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

end # module Srm0Accel
