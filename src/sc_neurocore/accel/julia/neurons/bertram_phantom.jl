# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia mirror of Bertram et al. 2000 phantom burster

module BertramPhantomAccel

export BertramPhantomState, step!, reset!, valid, simulate

mutable struct BertramPhantomState
    v::Float64
    n::Float64
    s1::Float64
    s2::Float64
    lambda_n::Float64
    g_ca::Float64
    g_k::Float64
    g_s1::Float64
    g_s2::Float64
    g_l::Float64
    e_ca::Float64
    e_k::Float64
    e_l::Float64
    c_m::Float64
    v_m::Float64
    s_m::Float64
    v_n::Float64
    s_n::Float64
    v_s1::Float64
    s_s1::Float64
    v_s2::Float64
    s_s2::Float64
    tau_n_bar::Float64
    tau_s1::Float64
    tau_s2::Float64
    dt::Float64
    v_threshold::Float64
end

BertramPhantomState() = BertramPhantomState(
    -43.0, 0.03, 0.1, 0.434, 1.1,
    280.0, 1300.0, 20.0, 32.0, 25.0,
    100.0, -80.0, -40.0, 4524.0,
    -22.0, 7.5, -9.0, 10.0, -40.0, 0.5, -42.0, 0.4,
    9.09, 1000.0, 120000.0, 0.5, -20.0,
)

function valid(state::BertramPhantomState)::Bool
    values = (
        state.v, state.n, state.s1, state.s2, state.lambda_n,
        state.g_ca, state.g_k, state.g_s1, state.g_s2, state.g_l,
        state.e_ca, state.e_k, state.e_l, state.c_m,
        state.v_m, state.s_m, state.v_n, state.s_n,
        state.v_s1, state.s_s1, state.v_s2, state.s_s2,
        state.tau_n_bar, state.tau_s1, state.tau_s2, state.dt, state.v_threshold,
    )
    all(isfinite, values) && -250.0 <= state.v <= 250.0 &&
        0.0 <= state.n <= 1.0 && 0.0 <= state.s1 <= 1.0 && 0.0 <= state.s2 <= 1.0 &&
        state.lambda_n > 0.0 && all(x -> x >= 0.0, (state.g_ca, state.g_k, state.g_s1, state.g_s2, state.g_l)) &&
        all(x -> x > 0.0, (state.c_m, state.s_m, state.s_n, state.s_s1, state.s_s2,
            state.tau_n_bar, state.tau_s1, state.tau_s2, state.dt))
end

_boltz(v::Float64, midpoint::Float64, slope::Float64) = 1.0 / (1.0 + exp((midpoint - v) / slope))

function _rhs(state::BertramPhantomState, values::NTuple{4, Float64}, current::Float64)
    v, n, s1, s2 = values
    m_inf = _boltz(v, state.v_m, state.s_m)
    n_inf = _boltz(v, state.v_n, state.s_n)
    s1_inf = _boltz(v, state.v_s1, state.s_s1)
    s2_inf = _boltz(v, state.v_s2, state.s_s2)
    tau_n = state.tau_n_bar / (1.0 + exp((v - state.v_n) / state.s_n))
    i_ca = state.g_ca * m_inf * (v - state.e_ca)
    i_k = state.g_k * n * (v - state.e_k)
    i_s1 = state.g_s1 * s1 * (v - state.e_k)
    i_s2 = state.g_s2 * s2 * (v - state.e_k)
    i_l = state.g_l * (v - state.e_l)
    (
        (-i_ca - i_k - i_s1 - i_s2 - i_l + current) / state.c_m,
        state.lambda_n * (n_inf - n) / tau_n,
        (s1_inf - s1) / state.tau_s1,
        (s2_inf - s2) / state.tau_s2,
    )
end

_shift(values, derivative, scale) = ntuple(i -> values[i] + scale * derivative[i], 4)

function step!(state::BertramPhantomState, current::Float64 = 0.0)::Int
    valid(state) && isfinite(current) || return -1
    previous_v = state.v
    values = (state.v, state.n, state.s1, state.s2)
    k1 = _rhs(state, values, current)
    k2 = _rhs(state, _shift(values, k1, 0.5 * state.dt), current)
    k3 = _rhs(state, _shift(values, k2, 0.5 * state.dt), current)
    k4 = _rhs(state, _shift(values, k3, state.dt), current)
    next = ntuple(i -> values[i] + state.dt * (k1[i] + 2k2[i] + 2k3[i] + k4[i]) / 6.0, 4)
    all(isfinite, next) && -250.0 <= next[1] <= 250.0 &&
        all(x -> -1e-9 <= x <= 1.0 + 1e-9, next[2:4]) || return -1
    state.v, state.n, state.s1, state.s2 = next[1], clamp(next[2], 0.0, 1.0),
        clamp(next[3], 0.0, 1.0), clamp(next[4], 0.0, 1.0)
    (state.v >= state.v_threshold && previous_v < state.v_threshold) ? 1 : 0
end

function reset!(state::BertramPhantomState)::Nothing
    state.v, state.n, state.s1, state.s2 = -43.0, 0.03, 0.1, 0.434
    nothing
end

function simulate(currents::AbstractVector{Float64}; state::BertramPhantomState = BertramPhantomState())
    voltages = Vector{Float64}(undef, length(currents))
    gates = Matrix{Float64}(undef, length(currents), 3)
    events = Vector{Int64}(undef, length(currents))
    for index in eachindex(currents)
        events[index] = step!(state, currents[index])
        voltages[index] = state.v
        gates[index, :] = (state.n, state.s1, state.s2)
    end
    (; voltages, gates, events, state)
end

end
