# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for expif

module ExpifAccel

export step!, simulate, ExpIFNeuronState

mutable struct ExpIFNeuronState
    v::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    v_rh::Float64
    delta_t::Float64
    tau::Float64
    dt::Float64
end

function ExpIFNeuronState()
    ExpIFNeuronState(-65.0, -65.0, -68.0, -50.0, -55.0, 2.0, 20.0, 0.1)
end

function step!(s::ExpIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        exp_term = s.delta_t * exp(clamp((s.v - s.v_rh) / s.delta_t, -20.0, 20.0))
        dv = (-(s.v - s.v_rest) + exp_term + I_ext) / s.tau * s.dt
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
    s = ExpIFNeuronState()
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

end # module ExpifAccel
