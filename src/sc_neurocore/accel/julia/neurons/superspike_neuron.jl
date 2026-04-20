# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for superspike_neuron

module SuperspikeNeuronAccel

export step!, simulate, SuperSpikeNeuronState

mutable struct SuperSpikeNeuronState
    v::Float64
    trace::Float64
    tau_m::Float64
    tau_e::Float64
    v_threshold::Float64
    v_reset::Float64
    beta_sg::Float64
    dt::Float64
    alpha_m::Float64
    alpha_e::Float64
end

function SuperSpikeNeuronState()
    SuperSpikeNeuronState(0.0, 0.0, 10.0, 10.0, 1.0, 0.0, 10.0, 1.0, 0.0, 0.0)
end

function surrogate_grad(s::SuperSpikeNeuronState)
    return 1.0 / (s.beta_sg * abs(s.v - s.v_threshold) + 1.0) ^ 2
end

function step!(s::SuperSpikeNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s.v = s.alpha_m * s.v + I_ext
        sg = s.surrogate_grad()
        s.trace = s.alpha_e * s.trace + sg
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
    s = SuperSpikeNeuronState()
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

end # module SuperspikeNeuronAccel
