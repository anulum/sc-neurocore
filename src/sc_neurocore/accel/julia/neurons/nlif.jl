# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for nlif

module NlifAccel

export step!, simulate, NonlinearLIFNeuronState

mutable struct NonlinearLIFNeuronState
    v::Float64
    w::Float64
    v_rest::Float64
    v_crit::Float64
    v_threshold::Float64
    v_reset::Float64
    a::Float64
    b::Float64
    tau_w::Float64
    c_m::Float64
    dt::Float64
end

function NonlinearLIFNeuronState()
    NonlinearLIFNeuronState(-65.0, 0.0, -65.0, -40.0, -20.0, -65.0, 0.04, 0.5, 100.0, 1.0, 0.1)
end

function step!(s::NonlinearLIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        cubic = s.a * (s.v - s.v_rest) * (s.v - s.v_crit)
        dv = (cubic - s.w + I_ext) / s.c_m * s.dt
        dw = (s.b * (s.v - s.v_rest) - s.w) / s.tau_w * s.dt
        s.v += dv
        s.w += dw
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
    s = NonlinearLIFNeuronState()
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

end # module NlifAccel
