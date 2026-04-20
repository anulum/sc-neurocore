# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for clif

module ClifAccel

export step!, simulate, ComplementaryLIFNeuronState

mutable struct ComplementaryLIFNeuronState
    v_pos::Float64
    v_neg::Float64
    tau::Float64
    v_threshold::Float64
    dt::Float64
    alpha::Float64
end

function ComplementaryLIFNeuronState()
    ComplementaryLIFNeuronState(0.0, 0.0, 10.0, 1.0, 1.0, 0.0)
end

function step!(s::ComplementaryLIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        inp_pos = max(I_ext, 0.0)
        inp_neg = max(-I_ext, 0.0)
        s.v_pos = s.alpha * s.v_pos + inp_pos
        s.v_neg = s.alpha * s.v_neg + inp_neg
        diff = s.v_pos - s.v_neg
        if diff >= s.v_threshold
            s.v_pos = 0.0
            s.v_neg = 0.0
            return 1
        end
        if diff <= -s.v_threshold
            s.v_pos = 0.0
            s.v_neg = 0.0
            return -1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = ComplementaryLIFNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v_pos
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module ClifAccel
