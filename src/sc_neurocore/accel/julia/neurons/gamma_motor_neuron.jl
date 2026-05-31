# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for gamma_motor_neuron

module GammaMotorNeuronAccel

export step!, simulate, GammaMotorNeuronState, static_type!

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

function static_type!(s::GammaMotorNeuronState)
    s.tau = 12.0
    s.tau_adapt = 200.0
    s.a_adapt = 0.5
    s.dynamic = 0.0
    validate!(s)
    return s
end

function validate!(s::GammaMotorNeuronState)
    values = (s.v, s.v_rest, s.v_reset, s.v_threshold, s.tau, s.adapt,
        s.tau_adapt, s.a_adapt, s.gain, s.dynamic, s.dt)
    all(isfinite, values) || throw(ArgumentError("gamma motor state and parameters must be finite"))
    s.tau > 0.0 || throw(ArgumentError("tau must be positive"))
    s.tau_adapt > 0.0 || throw(ArgumentError("tau_adapt must be positive"))
    s.dt > 0.0 || throw(ArgumentError("dt must be positive"))
    s.gain >= 0.0 || throw(ArgumentError("gain must be non-negative"))
    s.v_reset < s.v_threshold || throw(ArgumentError("v_reset must be below v_threshold"))
    return nothing
end

function step!(s::GammaMotorNeuronState, drive::Float64=0.0)
    validate!(s)
    isfinite(drive) || throw(ArgumentError("drive must be finite"))
    v_old = s.v
    adapt_old = s.adapt
    inp = s.gain * max(0.0, drive) - adapt_old
    v_target = s.v_rest + inp
    v_candidate = v_target + (v_old - v_target) * exp(-s.dt / s.tau)
    adapt_target = s.a_adapt * (v_candidate - s.v_rest)
    adapt_candidate = adapt_target + (adapt_old - adapt_target) * exp(-s.dt / s.tau_adapt)
    if !isfinite(v_candidate) || !isfinite(adapt_candidate)
        throw(ArgumentError("gamma motor candidate state must be finite"))
    end
    s.v = v_candidate
    s.adapt = adapt_candidate
    if s.v >= s.v_threshold
        s.v = s.v_reset
        return 1
    end
    return 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative"))
    s = GammaMotorNeuronState()
    s.dt = dt
    validate!(s)
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext)
        trace[t] = s.v
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module GammaMotorNeuronAccel
