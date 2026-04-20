# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for sfa

module SfaAccel

export step!, simulate, SFANeuronState

mutable struct SFANeuronState
    v::Float64
    g_sfa::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    tau_m::Float64
    tau_sfa::Float64
    delta_g::Float64
    e_k::Float64
    resistance::Float64
    dt::Float64
end

function SFANeuronState()
    SFANeuronState(-70.0, 0.0, -70.0, -70.0, -50.0, 10.0, 200.0, 0.5, -80.0, 1.0, 1.0)
end

function step!(s::SFANeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s.v += (-(s.v - s.v_rest) - s.g_sfa * (s.v - s.e_k) + s.resistance * I_ext) / s.tau_m * s.dt
        s.g_sfa *= exp(-s.dt / s.tau_sfa)
        if s.v >= s.v_threshold
            s.v = s.v_reset
            s.g_sfa += s.delta_g
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = SFANeuronState()
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

end # module SfaAccel
