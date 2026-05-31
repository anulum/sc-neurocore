# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for renshaw_cell

module RenshawCellAccel

export step!, simulate, RenshawCellState

mutable struct RenshawCellState
    v::Float64
    h::Float64
    n::Float64
    adapt::Float64
    g_na::Float64
    g_k::Float64
    g_adapt::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    c_m::Float64
    phi::Float64
    tau_adapt::Float64
    dt::Float64
    v_threshold::Float64
end

function RenshawCellState()
    RenshawCellState(-65.0, 0.8, 0.1, 0.0, 35.0, 9.0, 5.0, 0.12, 55.0, -90.0, -65.0, 1.0, 5.0, 50.0, 0.01, -20.0)
end

function _safe_rate(a::Float64, vhalf::Float64, v::Float64, k::Float64, fallback::Float64)
    d = v + vhalf
    if abs(d) < 1.0e-7
        return fallback
    end
    return a * d / (1.0 - exp(-d / k))
end

_clamp01(value::Float64) = min(1.0, max(0.0, value))
_probability(value::Float64) = isfinite(value) && 0.0 <= value <= 1.0
_voltage(value::Float64) = isfinite(value) && -150.0 <= value <= 100.0

function _exact_gate(previous::Float64, alpha::Float64, beta::Float64, phi::Float64, dt::Float64)
    total = phi * (alpha + beta)
    if !all(isfinite, (previous, alpha, beta, total, dt)) || total <= 0.0
        return nothing
    end
    steady = alpha / (alpha + beta)
    return _clamp01(steady + (previous - steady) * exp(-total * dt))
end

function _exact_relax(previous::Float64, steady::Float64, tau::Float64, dt::Float64)
    if !all(isfinite, (previous, steady, tau, dt)) || tau <= 0.0
        return nothing
    end
    return _clamp01(steady + (previous - steady) * exp(-dt / tau))
end

function _valid_state(s::RenshawCellState)
    return _voltage(s.v) &&
        _probability(s.h) &&
        _probability(s.n) &&
        _probability(s.adapt) &&
        all(isfinite, (s.g_na, s.g_k, s.g_adapt, s.g_l, s.e_na, s.e_k, s.e_l, s.c_m, s.phi, s.tau_adapt, s.dt, s.v_threshold)) &&
        s.g_na >= 0.0 &&
        s.g_k >= 0.0 &&
        s.g_adapt >= 0.0 &&
        s.g_l >= 0.0 &&
        s.c_m > 0.0 &&
        s.phi > 0.0 &&
        s.tau_adapt > 0.0 &&
        s.dt > 0.0
end

function step!(s::RenshawCellState, I_ext::Float64=0.0; dt::Float64=0.1)
    if !isfinite(I_ext) || !_valid_state(s)
        return 0
    end

    v_prev = s.v
    v = s.v
    h = s.h
    n = s.n
    adapt = s.adapt
    n_sub = max(1, Int(0.5 / max(s.dt, 0.001)))
    for _ in 1:n_sub
        am = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
        bm = 4.0 * exp(-(v + 60.0) / 18.0)
        m_inf = am / (am + bm)
        ah = 0.07 * exp(-(v + 58.0) / 20.0)
        bh = 1.0 / (1.0 + exp(-(v + 28.0) / 10.0))
        an = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
        bn = 0.125 * exp(-(v + 44.0) / 80.0)

        h_next = _exact_gate(h, ah, bh, s.phi, s.dt)
        n_next = _exact_gate(n, an, bn, s.phi, s.dt)
        if h_next === nothing || n_next === nothing
            return 0
        end
        adapt_inf = 1.0 / (1.0 + exp(-(v + 30.0) / 5.0))
        adapt_next = _exact_relax(adapt, adapt_inf, s.tau_adapt, s.dt)
        if adapt_next === nothing
            return 0
        end

        g_na = s.g_na * m_inf ^ 3 * h_next
        g_k = s.g_k * n_next ^ 4
        g_adapt = s.g_adapt * adapt_next
        g_total = g_na + g_k + g_adapt + s.g_l
        if !isfinite(g_total) || g_total <= 0.0
            return 0
        end
        steady_v = (I_ext + g_na * s.e_na + g_k * s.e_k + g_adapt * s.e_k + s.g_l * s.e_l) / g_total
        v_next = steady_v + (v - steady_v) * exp(-(g_total / s.c_m) * s.dt)
        if !_voltage(v_next) || !_probability(h_next) || !_probability(n_next) || !_probability(adapt_next)
            return 0
        end

        v = v_next
        h = h_next
        n = n_next
        adapt = adapt_next
    end

    s.v = v
    s.h = h
    s.n = n
    s.adapt = adapt
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = RenshawCellState()
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

end # module RenshawCellAccel
