# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for ltc

module LtcAccel

export step!, simulate, LiquidTimeConstantNeuronState

mutable struct LiquidTimeConstantNeuronState
    x::Float64
    tau_base::Float64
    w_tau::Float64
    w_x::Float64
    w_in::Float64
    bias::Float64
    v_threshold::Float64
    dt::Float64
end

function LiquidTimeConstantNeuronState()
    LiquidTimeConstantNeuronState(0.0, 10.0, -0.5, 0.8, 1.0, 0.0, 1.0, 1.0)
end

function step!(s::LiquidTimeConstantNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        tau = s.tau_base * (1.0 / (1.0 + exp(-(s.w_tau * I_ext + s.bias))))
        tau = max(tau, 0.1)
        f_target = tanh(s.w_x * s.x + s.w_in * I_ext)
        s.x += s.dt / tau * (-s.x + f_target)
        if s.x >= s.v_threshold
            s.x = 0.0
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = LiquidTimeConstantNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.x
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module LtcAccel
