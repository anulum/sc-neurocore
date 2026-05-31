# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for motor_unit

module MotorUnitAccel

export fast, slow, step!, simulate, MotorUnitState

mutable struct MotorUnitState
    v::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    tau_m::Float64
    adapt::Float64
    tau_adapt::Float64
    a_adapt::Float64
    gain::Float64
    force::Float64
    twitch_amp::Float64
    tau_twitch::Float64
    force_decay::Float64
    dt::Float64
end

function MotorUnitState()
    MotorUnitState(-65.0, -65.0, -70.0, -50.0, 10.0, 0.0, 100.0, 0.2, 1.0, 0.0, 0.05, 90.0, 0.0, 0.5)
end

function slow()
    return MotorUnitState()
end

function slow(s::MotorUnitState)
    s.v = -65.0
    s.v_rest = -65.0
    s.v_reset = -70.0
    s.v_threshold = -50.0
    s.tau_m = 10.0
    s.adapt = 0.0
    s.tau_adapt = 100.0
    s.a_adapt = 0.2
    s.gain = 1.0
    s.force = 0.0
    s.twitch_amp = 0.05
    s.tau_twitch = 90.0
    s.force_decay = 0.0
    s.dt = 0.5
    return s
end

function fast()
    return fast(MotorUnitState())
end

function fast(s::MotorUnitState)
    slow(s)
    s.tau_m = 6.0
    s.tau_adapt = 50.0
    s.a_adapt = 0.1
    s.twitch_amp = 0.3
    s.tau_twitch = 30.0
    return s
end

_voltage(value::Float64) = isfinite(value) && -150.0 <= value <= 100.0
_force(value::Float64) = isfinite(value) && 0.0 <= value <= 1.0

function _relax(previous::Float64, steady::Float64, tau::Float64, dt::Float64)
    if !all(isfinite, (previous, steady, tau, dt)) || tau <= 0.0 || dt <= 0.0
        return nothing
    end
    return steady + (previous - steady) * exp(-dt / tau)
end

function _valid_state(s::MotorUnitState)
    return _voltage(s.v) &&
        _voltage(s.v_rest) &&
        _voltage(s.v_reset) &&
        _voltage(s.v_threshold) &&
        _force(s.force) &&
        all(isfinite, (s.tau_m, s.adapt, s.tau_adapt, s.a_adapt, s.gain, s.twitch_amp, s.tau_twitch, s.force_decay, s.dt)) &&
        s.tau_m > 0.0 &&
        s.tau_adapt > 0.0 &&
        s.tau_twitch > 0.0 &&
        s.dt > 0.0 &&
        s.gain >= 0.0 &&
        s.twitch_amp >= 0.0 &&
        s.v_reset < s.v_threshold
end

function step!(s::MotorUnitState, I_ext::Float64=0.0; dt::Float64=0.1)
    if !isfinite(I_ext) || !_valid_state(s)
        return 0
    end

    force = s.force * exp(-s.dt / s.tau_twitch)
    input_drive = s.gain * max(0.0, I_ext) - s.adapt
    v_target = s.v_rest + input_drive
    v_candidate = _relax(s.v, v_target, s.tau_m, s.dt)
    if v_candidate === nothing || !_voltage(v_candidate)
        return 0
    end
    adapt_target = s.a_adapt * (v_candidate - s.v_rest)
    adapt_candidate = _relax(s.adapt, adapt_target, s.tau_adapt, s.dt)
    if adapt_candidate === nothing || !isfinite(adapt_candidate)
        return 0
    end

    spike = 0
    if v_candidate >= s.v_threshold
        v_candidate = s.v_reset
        force = min(1.0, force + s.twitch_amp)
        spike = 1
    end
    if !_voltage(v_candidate) || !_force(force)
        return 0
    end

    s.v = v_candidate
    s.adapt = adapt_candidate
    s.force = force
    return spike
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = MotorUnitState()
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

end # module MotorUnitAccel
