# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia implementation of the Brette et al. COBA LIF cell

module CobaLifAccel

export COBALIFNeuronState, reset!, simulate, simulate_trace, step!, valid

const V_MIN = -200.0
const V_MAX = 100.0
const G_MAX = 1.0e9

mutable struct COBALIFNeuronState
    v::Float64
    g_e::Float64
    g_i::Float64
    refractory_time::Float64
    c_m::Float64
    g_l::Float64
    e_l::Float64
    e_e::Float64
    e_i::Float64
    tau_e::Float64
    tau_i::Float64
    v_threshold::Float64
    v_reset::Float64
    refractory_period::Float64
    dt::Float64
end

function COBALIFNeuronState()
    return COBALIFNeuronState(
        -60.0,
        0.0,
        0.0,
        0.0,
        200.0,
        10.0,
        -60.0,
        0.0,
        -80.0,
        5.0,
        10.0,
        -50.0,
        -60.0,
        5.0,
        0.1,
    )
end

_finite(value::Float64)::Bool = isfinite(value)
_nonnegative(value::Float64)::Bool = isfinite(value) && value >= 0.0

function valid(s::COBALIFNeuronState)::Bool
    return _finite(s.v) && V_MIN <= s.v <= V_MAX &&
           _nonnegative(s.g_e) && s.g_e <= G_MAX &&
           _nonnegative(s.g_i) && s.g_i <= G_MAX &&
           _nonnegative(s.refractory_time) &&
           _finite(s.c_m) && s.c_m > 0.0 &&
           _nonnegative(s.g_l) &&
           all(isfinite, (s.e_l, s.e_e, s.e_i, s.v_threshold, s.v_reset)) &&
           V_MIN <= s.v_reset <= V_MAX &&
           _finite(s.tau_e) && s.tau_e > 0.0 &&
           _finite(s.tau_i) && s.tau_i > 0.0 &&
           _finite(s.refractory_period) && s.refractory_period > 0.0 &&
           s.refractory_time <= s.refractory_period &&
           _finite(s.dt) && s.dt > 0.0 && s.refractory_period >= s.dt
end

function _derivatives(
    s::COBALIFNeuronState,
    v::Float64,
    g_e::Float64,
    g_i::Float64,
    current::Float64,
)
    i_syn = g_e * (v - s.e_e) + g_i * (v - s.e_i)
    dv = (-s.g_l * (v - s.e_l) - i_syn + current) / s.c_m
    return dv, -g_e / s.tau_e, -g_i / s.tau_i
end

function _rk4_candidate(
    s::COBALIFNeuronState,
    v::Float64,
    g_e::Float64,
    g_i::Float64,
    current::Float64,
    dt::Float64,
)
    k1v, k1e, k1i = _derivatives(s, v, g_e, g_i, current)
    k2v, k2e, k2i = _derivatives(
        s,
        v + 0.5 * dt * k1v,
        g_e + 0.5 * dt * k1e,
        g_i + 0.5 * dt * k1i,
        current,
    )
    k3v, k3e, k3i = _derivatives(
        s,
        v + 0.5 * dt * k2v,
        g_e + 0.5 * dt * k2e,
        g_i + 0.5 * dt * k2i,
        current,
    )
    k4v, k4e, k4i = _derivatives(
        s,
        v + dt * k3v,
        g_e + dt * k3e,
        g_i + dt * k3i,
        current,
    )
    return (
        v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
        g_e + (dt / 6.0) * (k1e + 2.0 * k2e + 2.0 * k3e + k4e),
        g_i + (dt / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i),
    )
end

function _conductance_candidates(
    s::COBALIFNeuronState,
    g_e::Float64,
    g_i::Float64,
    dt::Float64,
)
    function decay(value::Float64, tau::Float64)::Float64
        k1 = -value / tau
        k2 = -(value + 0.5 * dt * k1) / tau
        k3 = -(value + 0.5 * dt * k2) / tau
        k4 = -(value + dt * k3) / tau
        return value + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    end
    return decay(g_e, s.tau_e), decay(g_i, s.tau_i)
end

function step!(
    s::COBALIFNeuronState,
    current::Float64=0.0;
    delta_ge::Float64=0.0,
    delta_gi::Float64=0.0,
    dt::Float64=s.dt,
)::Int
    if !valid(s) || !isfinite(current) || !_nonnegative(delta_ge) ||
       !_nonnegative(delta_gi) || !isfinite(dt) || dt <= 0.0 ||
       s.refractory_period < dt
        throw(DomainError(
            (s.v, s.g_e, s.g_i, s.refractory_time, current, delta_ge, delta_gi, dt),
            "COBA LIF state or step input lies outside the maintained contract",
        ))
    end
    g_e_pre = s.g_e + delta_ge
    g_i_pre = s.g_i + delta_gi
    if !isfinite(g_e_pre) || !isfinite(g_i_pre) || g_e_pre > G_MAX || g_i_pre > G_MAX
        throw(DomainError((g_e_pre, g_i_pre), "COBA LIF conductance candidate is invalid"))
    end

    next_v = s.v_reset
    next_g_e, next_g_i = _conductance_candidates(s, g_e_pre, g_i_pre, dt)
    next_refractory = s.refractory_time <= dt * (1.0 + 1.0e-12) ? 0.0 : s.refractory_time - dt
    spiked = false
    if s.refractory_time <= 0.0
        next_v, next_g_e, next_g_i = _rk4_candidate(s, s.v, g_e_pre, g_i_pre, current, dt)
        if !isfinite(next_v) || !(V_MIN <= next_v <= V_MAX)
            throw(DomainError(next_v, "COBA LIF voltage candidate left the safety envelope"))
        end
        next_refractory = 0.0
        spiked = next_v >= s.v_threshold
        if spiked
            next_v = s.v_reset
            next_refractory = s.refractory_period
        end
    end

    candidates = (next_v, next_g_e, next_g_i, next_refractory)
    if !all(isfinite, candidates) || !(V_MIN <= next_v <= V_MAX) ||
       next_g_e < 0.0 || next_g_i < 0.0 || next_refractory < 0.0
        throw(DomainError(candidates, "COBA LIF RK4 candidate left the safety envelope"))
    end
    s.v = next_v
    s.g_e = next_g_e
    s.g_i = next_g_i
    s.refractory_time = next_refractory
    s.dt = dt
    return spiked ? 1 : 0
end

function reset!(s::COBALIFNeuronState)::Nothing
    s.v = s.e_l
    s.g_e = 0.0
    s.g_i = 0.0
    s.refractory_time = 0.0
    return nothing
end

function simulate(
    n_steps::Int=1_000;
    current::Float64=500.0,
    delta_ge::Float64=0.0,
    delta_gi::Float64=0.0,
    dt::Float64=0.1,
)
    state = COBALIFNeuronState()
    state.dt = dt
    result = simulate_trace(
        state.v,
        state.g_e,
        state.g_i,
        state.refractory_time,
        state.c_m,
        state.g_l,
        state.e_l,
        state.e_e,
        state.e_i,
        state.tau_e,
        state.tau_i,
        state.v_threshold,
        state.v_reset,
        state.refractory_period,
        state.dt,
        n_steps,
        current,
        delta_ge,
        delta_gi,
    )
    return result.trace, result.spikes
end

function simulate_trace(
    v::Float64,
    g_e::Float64,
    g_i::Float64,
    refractory_time::Float64,
    c_m::Float64,
    g_l::Float64,
    e_l::Float64,
    e_e::Float64,
    e_i::Float64,
    tau_e::Float64,
    tau_i::Float64,
    v_threshold::Float64,
    v_reset::Float64,
    refractory_period::Float64,
    dt::Float64,
    n_steps::Int,
    current::Float64,
    delta_ge::Float64,
    delta_gi::Float64,
)
    if n_steps < 0
        throw(ArgumentError("COBA LIF n_steps must be non-negative"))
    end
    state = COBALIFNeuronState(
        v,
        g_e,
        g_i,
        refractory_time,
        c_m,
        g_l,
        e_l,
        e_e,
        e_i,
        tau_e,
        tau_i,
        v_threshold,
        v_reset,
        refractory_period,
        dt,
    )
    if !valid(state) || !isfinite(current) || !_nonnegative(delta_ge) ||
       !_nonnegative(delta_gi)
        throw(DomainError(
            (v, g_e, g_i, refractory_time, current, delta_ge, delta_gi),
            "COBA LIF simulation contract is invalid",
        ))
    end
    trace = zeros(Float64, n_steps)
    spikes = 0
    for index in eachindex(trace)
        spikes += step!(state, current; delta_ge=delta_ge, delta_gi=delta_gi)
        trace[index] = state.v
    end
    return (
        trace=trace,
        spikes=spikes,
        v_f=state.v,
        g_e_f=state.g_e,
        g_i_f=state.g_i,
        refractory_time_f=state.refractory_time,
    )
end

end # module CobaLifAccel
