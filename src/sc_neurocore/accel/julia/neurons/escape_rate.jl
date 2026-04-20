# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for escape_rate

module EscapeRateAccel

export step!, simulate, EscapeRateNeuronState

mutable struct EscapeRateNeuronState
    v::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    tau_m::Float64
    rho_0::Float64
    delta_u::Float64
    resistance::Float64
    dt::Float64
end

function EscapeRateNeuronState()
    EscapeRateNeuronState(-70.0, -70.0, -70.0, -50.0, 10.0, 0.001, 3.0, 1.0, 1.0)
end

function step!(s::EscapeRateNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        s.v += (-(s.v - s.v_rest) + s.resistance * I_ext) / s.tau_m * s.dt
        rate = s.rho_0 * safe_exp((s.v - s.v_threshold) / s.delta_u)
        p_spike = rate * s.dt
        if np.random.random() < p_spike
            s.v = s.v_reset
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = EscapeRateNeuronState()
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

end # module EscapeRateAccel
