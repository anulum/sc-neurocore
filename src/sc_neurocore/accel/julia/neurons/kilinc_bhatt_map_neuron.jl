# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for kilinc_bhatt_map_neuron

module KilincBhattMapNeuronAccel

export step!, simulate, KilincBhattMapNeuronState

mutable struct KilincBhattMapNeuronState
    x::Float64
    theta::Float64
    k::Float64
    beta::Float64
    gamma::Float64
    theta_spike::Float64
    x_threshold::Float64
end

function KilincBhattMapNeuronState()
    KilincBhattMapNeuronState(0.0, 0.0, 1.5, 0.95, 0.3, 0.8, 0.8)
end

function step!(s::KilincBhattMapNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        x_prev = s.x
        sig = 1.0 / (1.0 + exp(-(s.x - s.theta) * 4.0))
        x_new = -s.x + s.k * sig + I_ext
        spiked = (s.x >= s.theta_spike) ? 1.0 : 0.0
        theta_new = s.beta * s.theta + s.gamma * spiked
        s.x = max(-5.0, min(5.0, x_new))
        s.theta = max(-5.0, min(5.0, theta_new))
        if ! isfinite(s.x)
            s.x = 0.0
        end
        if ! isfinite(s.theta)
            s.theta = 0.0
        end
        return (s.x >= s.x_threshold && x_prev < s.x_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = KilincBhattMapNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.x
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module KilincBhattMapNeuronAccel
