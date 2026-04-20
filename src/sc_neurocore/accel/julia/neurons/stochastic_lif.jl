# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for stochastic_lif

module StochasticLifAccel

export step!, simulate, StochasticLIFState

mutable struct StochasticLIFState
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    tau_mem::Float64
    dt::Float64
    noise_std::Float64
    v::Float64
    ref_count::Int
    refractory_period::Int
end

function StochasticLIFState()
    StochasticLIFState(-65.0, -65.0, -50.0, 20.0, 0.1, 1.0, -65.0, 0, 3)
end

function step!(s::StochasticLIFState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        if s.ref_count > 0
            s.ref_count -= 1
            return 0
        end
        sqrt_dt = sqrt(s.dt)
        dv_det = (-(s.v - s.v_rest) + I_ext) / s.tau_mem * s.dt
        dv_noise = s.noise_std * sqrt_dt * randn()
        s.v += dv_det + dv_noise
        if s.v >= s.v_threshold
            s.v = s.v_reset
            s.ref_count = s.refractory_period
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = StochasticLIFState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module StochasticLifAccel
