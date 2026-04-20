# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for energy_lif

module EnergyLifAccel

export step!, simulate, EnergyLIFNeuronState

mutable struct EnergyLIFNeuronState
    v::Float64
    epsilon::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    tau_m::Float64
    tau_e::Float64
    alpha::Float64
    epsilon_0::Float64
    resistance::Float64
    dt::Float64
end

function EnergyLIFNeuronState()
    EnergyLIFNeuronState(-70.0, 1.0, -70.0, -70.0, -50.0, 10.0, 500.0, 0.1, 1.0, 1.0, 1.0)
end

function step!(s::EnergyLIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        effective_r = s.resistance * s.epsilon
        s.v += (-(s.v - s.v_rest) + effective_r * I_ext) / s.tau_m * s.dt
        s.epsilon += (s.epsilon_0 - s.epsilon) / s.tau_e * s.dt
        if s.v >= s.v_threshold && s.epsilon > 0.1
            s.v = s.v_reset
            s.epsilon -= s.alpha
            s.epsilon = max(0.0, s.epsilon)
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = EnergyLIFNeuronState()
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

end # module EnergyLifAccel
