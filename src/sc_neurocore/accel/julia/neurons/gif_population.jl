# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for gif_population

module GifPopulationAccel

export step!, simulate, GIFPopulationNeuronState

mutable struct GIFPopulationNeuronState
    v::Float64
    theta::Float64
    eta::Float64
    tau_m::Float64
    tau_eta::Float64
    delta_v::Float64
    lambda_0::Float64
    eta_increment::Float64
    v_rest::Float64
    v_reset::Float64
    dt::Float64
    _rng::Float64
end

function GIFPopulationNeuronState()
    GIFPopulationNeuronState(-65.0, -50.0, 0.0, 20.0, 100.0, 2.0, 0.001, 5.0, -65.0, -65.0, 0.5, 0.0)
end

function step!(s::GIFPopulationNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s.v += (-(s.v - s.v_rest) - s.eta + I_ext) / s.tau_m * s.dt
        s.eta *= exp(-s.dt / s.tau_eta)
        hazard = s.lambda_0 * exp(min((s.v - s.theta) / s.delta_v, 20.0))
        p_spike = 1.0 - exp(-hazard * s.dt)
        if s._rng.random() < p_spike
            s.v = s.v_reset
            s.eta += s.eta_increment
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = GIFPopulationNeuronState()
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

end # module GifPopulationAccel
