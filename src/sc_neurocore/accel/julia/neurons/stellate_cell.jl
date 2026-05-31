# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for stellate_cell

module StellateCellAccel

export step!, simulate, StellateCellState

mutable struct StellateCellState
    v::Float64
    h::Float64
    n::Float64
    p::Float64
    g_na::Float64
    g_k::Float64
    g_kv3::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    c_m::Float64
    phi::Float64
    dt::Float64
    v_threshold::Float64
    gain::Float64
    sub_steps::Int64
end

function StellateCellState()
    return StellateCellState(-65.0, 0.6, 0.32, 0.0, 35.0, 9.0, 3.0, 0.1, 55.0, -90.0, -65.0, 0.5, 5.0, 0.5, -20.0, 1.0, 50)
end

function _safe_exp(value::Float64)
    return exp(max(-60.0, min(60.0, value)))
end

function _safe_rate(a::Float64, vhalf::Float64, v::Float64, k::Float64, fallback::Float64)
    d = v + vhalf
    if abs(d) < 1e-7
        return fallback
    end
    z = -d / k
    if z > 60.0
        return 0.0
    elseif z < -60.0
        return a * d
    end
    return a * d / (1.0 - exp(z))
end

function _boltz(v::Float64, vh::Float64, k::Float64)
    z = -(v - vh) / k
    if z > 60.0
        return 0.0
    elseif z < -60.0
        return 1.0
    end
    return 1.0 / (1.0 + exp(z))
end

_clamp01(x::Float64) = max(0.0, min(1.0, x))

function _exact_relax(value::Float64, target::Float64, tau::Float64, dt::Float64)
    return target + (value - target) * exp(-dt / tau)
end

function _exact_hh_gate(value::Float64, alpha::Float64, beta::Float64, phi::Float64, dt::Float64)
    rate = phi * (alpha + beta)
    target = alpha / (alpha + beta)
    return target + (value - target) * exp(-rate * dt)
end

function _exact_voltage_step(v::Float64, input_current::Float64, c_m::Float64, dt::Float64, conductances)
    g_total = sum(pair[1] for pair in conductances)
    if g_total <= 0.0
        return v + dt * input_current / c_m
    end
    reversal_drive = sum(pair[1] * pair[2] for pair in conductances)
    v_inf = (input_current + reversal_drive) / g_total
    return v_inf + (v - v_inf) * exp(-dt * g_total / c_m)
end

function _validate(s::StellateCellState)
    finite_values = (
        s.v, s.h, s.n, s.p, s.g_na, s.g_k, s.g_kv3, s.g_l,
        s.e_na, s.e_k, s.e_l, s.c_m, s.phi, s.dt, s.v_threshold, s.gain,
    )
    all(isfinite, finite_values) || throw(ArgumentError("stellate cell state and parameters must be finite"))
    -100.0 <= s.v <= 60.0 || throw(ArgumentError("stellate cell v must stay in [-100, 60]"))
    all(x -> 0.0 <= x <= 1.0, (s.h, s.n, s.p)) ||
        throw(ArgumentError("stellate cell gates must stay in [0, 1]"))
    all(x -> x >= 0.0, (s.g_na, s.g_k, s.g_kv3, s.g_l)) ||
        throw(ArgumentError("stellate cell conductances must be non-negative"))
    (s.c_m > 0.0 && s.phi > 0.0 && s.dt > 0.0) ||
        throw(ArgumentError("stellate cell capacitance, rate scale, and timestep must be positive"))
    s.sub_steps > 0 || throw(ArgumentError("stellate cell sub-step count must be positive"))
    s.gain >= 0.0 || throw(ArgumentError("stellate cell gain must be non-negative"))
    return nothing
end

function step!(s::StellateCellState, I_ext::Float64=0.0; dt::Float64=s.dt)
    _validate(s)
    isfinite(I_ext) || throw(ArgumentError("current must be finite"))
    (isfinite(dt) && dt > 0.0) || throw(ArgumentError("dt must be finite and positive"))

    inp = s.gain * I_ext
    sub_dt = dt / Float64(s.sub_steps)
    fired = 0
    v = s.v
    h = s.h
    n = s.n
    p = s.p

    for _ in 1:s.sub_steps
        alpha_m = _safe_rate(0.1, 35.0, v, 10.0, 1.0)
        beta_m = 4.0 * _safe_exp(-(v + 60.0) / 18.0)
        m_inf = alpha_m / (alpha_m + beta_m)
        alpha_h = 0.07 * _safe_exp(-(v + 58.0) / 20.0)
        beta_h = _boltz(v, -28.0, 10.0)
        alpha_n = _safe_rate(0.01, 34.0, v, 10.0, 0.1)
        beta_n = 0.125 * _safe_exp(-(v + 44.0) / 80.0)
        p_inf = _boltz(v, -10.0, 10.0)
        tau_p = 1.0 + 4.0 / (1.0 + _safe_exp((v + 20.0) / 15.0))

        h = _clamp01(_exact_hh_gate(h, alpha_h, beta_h, s.phi, sub_dt))
        n = _clamp01(_exact_hh_gate(n, alpha_n, beta_n, s.phi, sub_dt))
        p = _clamp01(_exact_relax(p, p_inf, tau_p, sub_dt))

        g_na_eff = s.g_na * m_inf^3 * h
        g_k_eff = s.g_k * n^4
        g_kv3_eff = s.g_kv3 * p^2
        v = max(-100.0, min(60.0, _exact_voltage_step(v, inp, s.c_m, sub_dt, (
            (g_na_eff, s.e_na),
            (g_k_eff, s.e_k),
            (g_kv3_eff, s.e_k),
            (s.g_l, s.e_l),
        ))))
        all(isfinite, (v, h, n, p)) || throw(ArgumentError("stellate cell integration produced non-finite state"))
        if v >= s.v_threshold
            fired = 1
            v = -65.0
        end
    end

    s.v = v
    s.h = h
    s.n = n
    s.p = p
    return fired
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.5)
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative"))
    s = StellateCellState()
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

end # module StellateCellAccel
