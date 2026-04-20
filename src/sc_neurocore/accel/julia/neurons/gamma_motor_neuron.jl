# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for gamma_motor_neuron

module GammaMotorNeuronAccel

export step!, simulate, GammaMotorNeuronState

mutable struct GammaMotorNeuronState
    v::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    tau::Float64
    adapt::Float64
    tau_adapt::Float64
    a_adapt::Float64
    gain::Float64
    dynamic::Float64
    dt::Float64
end

function GammaMotorNeuronState()
    GammaMotorNeuronState(-65.0, -65.0, -70.0, -50.0, 8.0, 0.0, 100.0, 0.3, 1.0, 1.0, 0.5)
end

function static_type(s::GammaMotorNeuronState)
    return cls(tau=12.0, tau_adapt=200.0, a_adapt=0.5, dynamic=false)
end

function step!(s::GammaMotorNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        inp = s.gain * max(0.0, drive) - s.adapt
        s.v += (-(s.v - s.v_rest) + inp) / s.tau * s.dt
        s.adapt += (s.a_adapt * (s.v - s.v_rest) - s.adapt) / s.tau_adapt * s.dt
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
    s = GammaMotorNeuronState()
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

end # module GammaMotorNeuronAccel
