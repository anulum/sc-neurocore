# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for hodgkin_huxley

module HodgkinHuxleyAccel

export step!, simulate, validate_hodgkin_huxley, HodgkinHuxleyNeuronState

mutable struct HodgkinHuxleyNeuronState
    v::Float64
    m::Float64
    h::Float64
    n::Float64
    c_m::Float64
    g_na::Float64
    g_k::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    dt::Float64
    v_threshold::Float64
end

function HodgkinHuxleyNeuronState()
    HodgkinHuxleyNeuronState(-65.0, 0.05, 0.6, 0.32, 1.0, 120.0, 36.0, 0.3, 50.0, -77.0, -54.4, 0.01, 0.0)
end

@inline _finite(xs...) = all(isfinite, xs)

@inline function _safe_exp(x::Float64)
    (!_finite(x) || x > 700.0) && return nothing
    return exp(x)
end

# Singular-limit opening rate scale*d/(1 - exp(-d/denom)) with d = v + shift, returning the
# analytic limit `limit` when |d| < 1e-7 (bit-for-bit the guard in models/hodgkin_huxley.py).
@inline function _opening_rate(scale::Float64, shift::Float64, denom::Float64, limit::Float64, v::Float64)
    d = v + shift
    abs(d) < 1e-7 && return limit
    e = _safe_exp(-d / denom)
    e === nothing && return nothing
    value = scale * d / (1.0 - e)
    return isfinite(value) ? value : nothing
end

_alpha_m(v::Float64) = _opening_rate(0.1, 40.0, 10.0, 1.0, v)

function _beta_m(v::Float64)
    e = _safe_exp(-(v + 65.0) / 18.0)
    e === nothing && return nothing
    return 4.0 * e
end

function _alpha_h(v::Float64)
    e = _safe_exp(-(v + 65.0) / 20.0)
    e === nothing && return nothing
    return 0.07 * e
end

function _beta_h(v::Float64)
    e = _safe_exp(-(v + 35.0) / 10.0)
    e === nothing && return nothing
    return 1.0 / (1.0 + e)
end

_alpha_n(v::Float64) = _opening_rate(0.01, 55.0, 10.0, 0.1, v)

function _beta_n(v::Float64)
    e = _safe_exp(-(v + 65.0) / 80.0)
    e === nothing && return nothing
    return 0.125 * e
end

function _valid_state(v::Float64, m::Float64, h::Float64, n::Float64)::Bool
    return _finite(v, m, h, n) &&
        -250.0 <= v <= 250.0 &&
        -0.05 <= m <= 1.05 && -0.05 <= h <= 1.05 && -0.05 <= n <= 1.05
end

function validate_hodgkin_huxley(s::HodgkinHuxleyNeuronState)::Bool
    return _finite(s.v, s.m, s.h, s.n, s.c_m, s.g_na, s.g_k, s.g_l, s.e_na, s.e_k, s.e_l, s.dt, s.v_threshold) &&
        s.g_na >= 0.0 && s.g_k >= 0.0 && s.g_l >= 0.0 && s.c_m > 0.0 && s.dt > 0.0 &&
        _valid_state(s.v, s.m, s.h, s.n)
end

# One macro step over round(1/dt) explicit-Euler sub-steps: gates advance first, then the
# membrane voltage uses the freshly-updated gates (the models/hodgkin_huxley.py baseline_euler
# order). Fail-closed — returns nothing (state untouched) on any non-finite intermediate.
function _euler_candidate(s::HodgkinHuxleyNeuronState, I_ext::Float64)
    (!validate_hodgkin_huxley(s) || !isfinite(I_ext)) && return nothing
    v, m, h, n = s.v, s.m, s.h, s.n
    substeps = Int(round(1.0 / s.dt))
    for _ in 1:substeps
        am = _alpha_m(v)
        am === nothing && return nothing
        bm = _beta_m(v)
        bm === nothing && return nothing
        ah = _alpha_h(v)
        ah === nothing && return nothing
        bh = _beta_h(v)
        bh === nothing && return nothing
        an = _alpha_n(v)
        an === nothing && return nothing
        bn = _beta_n(v)
        bn === nothing && return nothing
        m += (am * (1.0 - m) - bm * m) * s.dt
        h += (ah * (1.0 - h) - bh * h) * s.dt
        n += (an * (1.0 - n) - bn * n) * s.dt
        i_na = s.g_na * m^3 * h * (v - s.e_na)
        i_k = s.g_k * n^4 * (v - s.e_k)
        i_l = s.g_l * (v - s.e_l)
        v += (-i_na - i_k - i_l + I_ext) / s.c_m * s.dt
        _valid_state(v, m, h, n) || return nothing
    end
    return (v, m, h, n)
end

function step!(s::HodgkinHuxleyNeuronState, I_ext::Float64=0.0; dt::Union{Nothing,Float64}=nothing)
    if dt !== nothing
        if !isfinite(dt) || dt <= 0.0
            return -1
        end
        s.dt = dt
    end
    v_prev = s.v
    candidate = _euler_candidate(s, I_ext)
    candidate === nothing && return -1
    s.v, s.m, s.h, s.n = candidate
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.01)
    s = HodgkinHuxleyNeuronState()
    s.dt = dt
    trace = zeros(Float64, n_steps)
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

end # module HodgkinHuxleyAccel
