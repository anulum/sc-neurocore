# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for connor_stevens

module ConnorStevensAccel

export step!, simulate, validate_connor_stevens, ConnorStevensNeuronState

mutable struct ConnorStevensNeuronState
    v::Float64
    m::Float64
    h::Float64
    n::Float64
    a::Float64
    b::Float64
    g_na::Float64
    g_k::Float64
    g_a::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_a::Float64
    e_l::Float64
    c_m::Float64
    dt::Float64
    v_threshold::Float64
end

function ConnorStevensNeuronState()
    ConnorStevensNeuronState(-68.0, 0.01, 0.99, 0.1, 0.5, 0.1, 120.0, 20.0, 47.7, 0.3, 55.0, -72.0, -75.0, -17.0, 1.0, 0.01, 0.0)
end

@inline _finite(xs...) = all(isfinite, xs)

@inline function _safe_exp(x::Float64)
    (!_finite(x) || x > 700.0) && return nothing
    return exp(x)
end

@inline function _safe_rate(scale::Float64, shift::Float64, v::Float64, denom::Float64)
    delta = v + shift
    x = delta / denom
    abs(x) < 1e-9 && return scale * denom
    e = _safe_exp(-x)
    e === nothing && return nothing
    value = scale * delta / (1.0 - e)
    return isfinite(value) ? value : nothing
end

function validate_connor_stevens(s::ConnorStevensNeuronState)::Bool
    return _finite(s.v, s.m, s.h, s.n, s.a, s.b, s.g_na, s.g_k, s.g_a, s.g_l, s.e_na, s.e_k, s.e_a, s.e_l, s.c_m, s.dt, s.v_threshold) &&
        s.g_na >= 0.0 && s.g_k >= 0.0 && s.g_a >= 0.0 && s.g_l >= 0.0 && s.c_m > 0.0 && s.dt > 0.0 &&
        _candidate_valid((s.v, s.m, s.h, s.n, s.a, s.b))
end

function _candidate_valid(state::NTuple{6,Float64})::Bool
    v, m, h, n, a, b = state
    return _finite(v, m, h, n, a, b) &&
        -250.0 <= v <= 250.0 &&
        -0.05 <= m <= 1.05 && -0.05 <= h <= 1.05 && -0.05 <= n <= 1.05 &&
        -0.05 <= a <= 1.5 && -0.05 <= b <= 1.05
end

function _derivatives(s::ConnorStevensNeuronState, state::NTuple{6,Float64}, I_ext::Float64)
    v, m, h, n, a, b = state
    am = _safe_rate(0.38, 29.7, v, 10.0)
    bm_exp = _safe_exp(-(v + 54.7) / 18.0)
    ah_exp = _safe_exp(-(v + 48.0) / 20.0)
    bh_exp = _safe_exp(-(v + 18.0) / 10.0)
    an = _safe_rate(0.02, 45.7, v, 10.0)
    bn_exp = _safe_exp(-(v + 55.7) / 80.0)
    a_num = _safe_exp((v + 94.22) / 31.84)
    a_den = _safe_exp((v + 1.17) / 28.93)
    tau_a_exp = _safe_exp((v + 55.96) / 20.12)
    b_exp = _safe_exp((v + 53.3) / 14.54)
    tau_b_exp = _safe_exp((v + 50.0) / 16.027)
    any(x -> x === nothing, (am, bm_exp, ah_exp, bh_exp, an, bn_exp, a_num, a_den, tau_a_exp, b_exp, tau_b_exp)) && return nothing

    a_base = 0.0761 * a_num / (1.0 + a_den)
    (a_base < 0.0 || !isfinite(a_base)) && return nothing
    a_inf = a_base^(1.0 / 3.0)
    tau_a = 0.3632 + 1.158 / (1.0 + tau_a_exp)
    b_base = 1.0 / (1.0 + b_exp)
    b_inf = b_base^4
    tau_b = 1.24 + 2.678 / (1.0 + tau_b_exp)
    i_na = s.g_na * m^3 * h * (v - s.e_na)
    i_k = s.g_k * n^4 * (v - s.e_k)
    i_a = s.g_a * a^3 * b * (v - s.e_a)
    i_l = s.g_l * (v - s.e_l)
    deriv = (
        (-i_na - i_k - i_a - i_l + I_ext) / s.c_m,
        am * (1.0 - m) - bm_exp * 15.2 * m,
        ah_exp * 0.266 * (1.0 - h) - 3.8 / (1.0 + bh_exp) * h,
        an * (1.0 - n) - bn_exp * 0.25 * n,
        (a_inf - a) / tau_a,
        (b_inf - b) / tau_b,
    )
    return all(isfinite, deriv) ? deriv : nothing
end

function _rk4_candidate(s::ConnorStevensNeuronState, I_ext::Float64)
    (!validate_connor_stevens(s) || !isfinite(I_ext)) && return nothing
    state = (s.v, s.m, s.h, s.n, s.a, s.b)
    substeps = Int(floor(1.0 / max(s.dt, 0.001)))
    for _ in 1:substeps
        k1 = _derivatives(s, state, I_ext)
        k1 === nothing && return nothing
        k2_state = ntuple(i -> state[i] + 0.5 * s.dt * k1[i], 6)
        k2 = _derivatives(s, k2_state, I_ext)
        k2 === nothing && return nothing
        k3_state = ntuple(i -> state[i] + 0.5 * s.dt * k2[i], 6)
        k3 = _derivatives(s, k3_state, I_ext)
        k3 === nothing && return nothing
        k4_state = ntuple(i -> state[i] + s.dt * k3[i], 6)
        k4 = _derivatives(s, k4_state, I_ext)
        k4 === nothing && return nothing
        state = ntuple(i -> state[i] + s.dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0, 6)
        _candidate_valid(state) || return nothing
    end
    return state
end

function step!(s::ConnorStevensNeuronState, I_ext::Float64=0.0; dt::Union{Nothing,Float64}=nothing)
    if dt !== nothing
        if !isfinite(dt) || dt <= 0.0
            return -1
        end
        s.dt = dt
    end
    v_prev = s.v
    candidate = _rk4_candidate(s, I_ext)
    candidate === nothing && return -1
    s.v, s.m, s.h, s.n, s.a, s.b = candidate
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.01)
    s = ConnorStevensNeuronState()
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

end # module ConnorStevensAccel
