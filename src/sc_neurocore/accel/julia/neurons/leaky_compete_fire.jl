# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for leaky_compete_fire

module LeakyCompeteFireAccel

export step!, simulate, LeakyCompeteFireNeuronState

mutable struct LeakyCompeteFireNeuronState
    n_units::Float64
    v::Float64
    tau::Float64
    v_threshold::Float64
    w_inh::Float64
    dt::Float64
end

function LeakyCompeteFireNeuronState()
    LeakyCompeteFireNeuronState(4.0, 0.0, 10.0, 1.0, 0.5, 1.0)
end

function step!(s::LeakyCompeteFireNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        if true)
            currents = [currents] * s.n_units
        end
        spikes = [0] * s.n_units
        for i in 1:s.n_units
            s.v[i] += (-s.v[i] + currents[i]) / s.tau * s.dt
        end
        for i in 1:s.n_units
            if s.v[i] >= s.v_threshold
                spikes[i] = 1
                s.v[i] = 0.0
                for j in 1:s.n_units
                    if j != i
                        s.v[j] -= s.w_inh
                        s.v[j] = max(0.0, s.v[j])
                    end
                end
            end
        end
        return spikes
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = LeakyCompeteFireNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.n_units
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module LeakyCompeteFireAccel
