# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for sigmoid_rate

module SigmoidRateAccel

export step!, simulate, SigmoidRateNeuronState

mutable struct SigmoidRateNeuronState
    r::Float64
    tau::Float64
    beta::Float64
    theta::Float64
    dt::Float64
end

function SigmoidRateNeuronState()
    SigmoidRateNeuronState(0.0, 10.0, 1.0, 0.0, 0.1)
end

function step!(s::SigmoidRateNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        sigma = 1.0 / (1.0 + exp(-s.beta * (I_ext - s.theta)))
        s.r += (-s.r + sigma) / s.tau * s.dt
        return s.r
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = SigmoidRateNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.r
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module SigmoidRateAccel
