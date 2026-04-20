# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for spike_response

module SpikeResponseAccel

export step!, simulate, SpikeResponseNeuronState

mutable struct SpikeResponseNeuronState
    v::Float64
    v_threshold::Float64
    tau_eta::Float64
    tau_kappa::Float64
    eta_reset::Float64
    time_since_spike::Float64
    dt::Float64
end

function SpikeResponseNeuronState()
    SpikeResponseNeuronState(0.0, 1.0, 10.0, 5.0, -5.0, 1000.0, 1.0)
end

function step!(s::SpikeResponseNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        eta = (s.time_since_spike < 100.0) ? s.eta_reset * exp(-s.time_since_spike / s.tau_eta) : 0.0
        kappa = weighted_input * (1.0 - exp(-s.dt / s.tau_kappa))
        s.v = eta + kappa
        s.time_since_spike += s.dt
        if s.v >= s.v_threshold
            s.time_since_spike = 0.0
            s.v = 0.0
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = SpikeResponseNeuronState()
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

end # module SpikeResponseAccel
