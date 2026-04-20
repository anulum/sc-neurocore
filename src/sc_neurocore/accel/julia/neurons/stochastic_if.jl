# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for stochastic_if

module StochasticIfAccel

export step!, simulate, StochasticIFNeuronState

mutable struct StochasticIFNeuronState
    v::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    tau_m::Float64
    mu::Float64
    sigma::Float64
    dt::Float64
end

function StochasticIFNeuronState()
    StochasticIFNeuronState(-70.0, -70.0, -70.0, -50.0, 20.0, 0.0, 3.0, 1.0)
end

function step!(s::StochasticIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        noise = s.sigma * sqrt(s.dt / s.tau_m) * randn()
        s.v += (-(s.v - s.v_rest) + s.mu + I_ext) / s.tau_m * s.dt + noise
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
    s = StochasticIFNeuronState()
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

end # module StochasticIfAccel
