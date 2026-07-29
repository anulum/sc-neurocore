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

function valid(s::KilincBhattMapNeuronState)
    isfinite(s.x) && -5.0 <= s.x <= 5.0 &&
        isfinite(s.theta) && -5.0 <= s.theta <= 5.0 &&
        isfinite(s.k) && 0.0 <= s.k <= 5.0 &&
        isfinite(s.beta) && 0.0 <= s.beta <= 1.0 &&
        isfinite(s.gamma) && 0.0 <= s.gamma <= 2.0 &&
        isfinite(s.theta_spike) && 0.0 <= s.theta_spike <= 2.0 &&
        isfinite(s.x_threshold) && 0.0 <= s.x_threshold <= 2.0
end

function sigmoid(z::Float64)
    if z >= 0.0
        return 1.0 / (1.0 + exp(-z))
    end
    exp_z = exp(z)
    exp_z / (1.0 + exp_z)
end

function step!(s::KilincBhattMapNeuronState, I_ext::Float64=0.0; dt::Float64=1.0)
    isfinite(I_ext) || throw(ArgumentError("current must be finite"))
    dt == 1.0 || throw(ArgumentError("Kilinc-Bhatt is a discrete map and requires dt=1"))
    valid(s) || throw(ArgumentError("Kilinc-Bhatt state and parameters must satisfy the public bounds"))

    x_prev = s.x
    sig = sigmoid((s.x - s.theta) * 4.0)
    x_new = -s.x + s.k * sig + I_ext
    spiked = (s.x >= s.theta_spike) ? 1.0 : 0.0
    theta_new = s.beta * s.theta + s.gamma * spiked
    if !isfinite(x_new) || !isfinite(theta_new)
        throw(DomainError((x_new, theta_new), "Kilinc-Bhatt candidate state became non-finite"))
    end

    s.x = clamp(x_new, -5.0, 5.0)
    s.theta = clamp(theta_new, -5.0, 5.0)
    (s.x >= s.x_threshold && x_prev < s.x_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=1.0)
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative"))
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
