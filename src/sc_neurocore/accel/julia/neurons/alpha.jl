# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for alpha

module AlphaAccel

export step!, simulate, AlphaNeuronState

mutable struct AlphaNeuronState
    v::Float64
    i_exc::Float64
    i_inh::Float64
    v_rest::Float64
    v_threshold::Float64
    tau_v::Float64
    tau_exc::Float64
    tau_inh::Float64
    dt::Float64
end

function AlphaNeuronState()
    AlphaNeuronState(0.0, 0.0, 0.0, 0.0, 1.0, 20.0, 5.0, 10.0, 1.0)
end

function step!(s::AlphaNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s.i_exc += (-s.i_exc / s.tau_exc + exc_current) * s.dt
        s.i_inh += (-s.i_inh / s.tau_inh + inh_current) * s.dt
        dv = (-(s.v - s.v_rest) + s.i_exc - s.i_inh) / s.tau_v * s.dt
        s.v += dv
        if s.v >= s.v_threshold
            s.v = s.v_rest
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = AlphaNeuronState()
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

end # module AlphaAccel
