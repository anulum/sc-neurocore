# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for sigma_delta

module SigmaDeltaAccel

export step!, simulate, SigmaDeltaNeuronState

mutable struct SigmaDeltaNeuronState
    sigma::Float64
    v_threshold::Float64
end

function SigmaDeltaNeuronState()
    SigmaDeltaNeuronState(0.0, 1.0)
end

function step!(s::SigmaDeltaNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s.sigma += I_ext
        if s.sigma >= s.v_threshold
            s.sigma -= s.v_threshold
            return 1
        elseif s.sigma <= -s.v_threshold
            s.sigma += s.v_threshold
            return -1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = SigmaDeltaNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.sigma
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module SigmaDeltaAccel
