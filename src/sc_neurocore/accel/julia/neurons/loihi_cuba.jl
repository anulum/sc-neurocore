# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for loihi_cuba

module LoihiCubaAccel

export step!, simulate, LoihiCUBANeuronState

mutable struct LoihiCUBANeuronState
    v::Float64
    u::Float64
    tau_v::Float64
    tau_u::Float64
    v_threshold::Float64
    v_reset::Float64
end

function LoihiCUBANeuronState()
    LoihiCUBANeuronState(0.0, 0.0, 10.0, 5.0, 1000.0, 0.0)
end

function step!(s::LoihiCUBANeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s.u = s.u - s.u // s.tau_u + weighted_input
        s.v = s.v - s.v // s.tau_v + s.u
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
    s = LoihiCUBANeuronState()
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

end # module LoihiCubaAccel
