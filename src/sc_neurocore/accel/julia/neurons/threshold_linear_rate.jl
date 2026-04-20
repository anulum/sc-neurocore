# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for threshold_linear_rate

module ThresholdLinearRateAccel

export step!, simulate, ThresholdLinearRateNeuronState

mutable struct ThresholdLinearRateNeuronState
    r::Float64
    theta::Float64
    gain::Float64
end

function ThresholdLinearRateNeuronState()
    ThresholdLinearRateNeuronState(0.0, 0.0, 1.0)
end

function step!(s::ThresholdLinearRateNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s.r = s.gain * max(0.0, I_ext - s.theta)
        return s.r
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = ThresholdLinearRateNeuronState()
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

end # module ThresholdLinearRateAccel
