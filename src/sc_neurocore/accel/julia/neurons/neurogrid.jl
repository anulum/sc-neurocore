# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for neurogrid

module NeurogridAccel

export step!, simulate, NeuroGridNeuronState

mutable struct NeuroGridNeuronState
    v_s::Float64
    v_d::Float64
    tau_s::Float64
    tau_d::Float64
    g_c::Float64
    delta_t::Float64
    v_rest::Float64
    v_threshold::Float64
    v_peak::Float64
    v_reset::Float64
    dt::Float64
end

function NeuroGridNeuronState()
    NeuroGridNeuronState(-65.0, -65.0, 20.0, 50.0, 0.5, 2.0, -65.0, -50.0, 20.0, -65.0, 0.1)
end

function step!(s::NeuroGridNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        dv_d = (-(s.v_d - s.v_rest) + I_ext - s.g_c * (s.v_d - s.v_s)) / s.tau_d
        s.v_d += dv_d * s.dt
        exp_term = s.delta_t * exp(min((s.v_s - s.v_threshold) / s.delta_t, 20.0))
        dv_s = (-(s.v_s - s.v_rest) + exp_term + s.g_c * (s.v_d - s.v_s)) / s.tau_s
        s.v_s += dv_s * s.dt
        if s.v_s >= s.v_peak
            s.v_s = s.v_reset
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = NeuroGridNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v_s
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module NeurogridAccel
