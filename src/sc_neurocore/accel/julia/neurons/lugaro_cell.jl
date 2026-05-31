# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for lugaro_cell

module LugaroCellAccel

export step!, simulate, with_serotonin, LugaroCellState

mutable struct LugaroCellState
    v::Float64
    adapt::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    tau_m::Float64
    tau_adapt::Float64
    a_adapt::Float64
    gain::Float64
    serotonin::Float64
    dt::Float64
end

function LugaroCellState()
    return LugaroCellState(-55.0, 0.0, -55.0, -65.0, -48.0, 10.0, 150.0, 0.05, 2.0, 0.0, 0.5)
end

function with_serotonin(level::Float64)
    s = LugaroCellState()
    s.serotonin = max(0.0, min(1.0, level))
    return s
end

function _validate(s::LugaroCellState)
    finite_values = (
        s.v, s.adapt, s.v_rest, s.v_reset, s.v_threshold, s.tau_m, s.tau_adapt,
        s.a_adapt, s.gain, s.serotonin, s.dt,
    )
    all(isfinite, finite_values) || throw(ArgumentError("lugaro cell state and parameters must be finite"))
    (s.tau_m > 0.0 && s.tau_adapt > 0.0 && s.dt > 0.0) ||
        throw(ArgumentError("lugaro cell time constants and timestep must be positive"))
    s.a_adapt >= 0.0 || throw(ArgumentError("lugaro cell adaptation coupling must be non-negative"))
    s.gain >= 0.0 || throw(ArgumentError("lugaro cell gain must be non-negative"))
    0.0 <= s.serotonin <= 1.0 || throw(ArgumentError("lugaro cell serotonin must stay in [0, 1]"))
    s.adapt >= 0.0 || throw(ArgumentError("lugaro cell adaptation current must be non-negative"))
    (s.v_threshold > s.v_reset && s.v_threshold > s.v_rest) ||
        throw(ArgumentError("lugaro cell threshold must exceed reset and rest potentials"))
    return nothing
end

function step!(s::LugaroCellState, I_ext::Float64=0.0; dt::Float64=s.dt)
    _validate(s)
    isfinite(I_ext) || throw(ArgumentError("current must be finite"))
    (isfinite(dt) && dt > 0.0) || throw(ArgumentError("dt must be finite and positive"))

    effective_gain = s.gain * (1.0 + 0.5 * s.serotonin)
    inp = effective_gain * I_ext
    v_inf = s.v_rest + inp - s.adapt
    v_next = v_inf + (s.v - v_inf) * exp(-dt / s.tau_m)
    adapt_inf = max(0.0, s.a_adapt * max(0.0, v_next - s.v_rest))
    adapt_next = adapt_inf + (s.adapt - adapt_inf) * exp(-dt / s.tau_adapt)
    adapt_next = max(0.0, adapt_next)
    all(isfinite, (v_next, adapt_next)) ||
        throw(ArgumentError("lugaro cell integration produced non-finite state"))

    if v_next >= s.v_threshold
        s.v = s.v_reset
        s.adapt = adapt_next + 1.0
        return 1
    end

    s.v = max(-100.0, min(60.0, v_next))
    s.adapt = adapt_next
    return 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.5)
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative"))
    s = LugaroCellState()
    s.dt = dt
    _validate(s)
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext)
        trace[t] = s.v
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module LugaroCellAccel
