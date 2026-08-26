# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia mirror for MainenSejnowskiNeuron

module MainenSejnowskiAccel

export step!, simulate, reset!, valid, MainenSejnowskiNeuronState

"""Complete two-compartment soma+axon state mirroring the Python reference."""
mutable struct MainenSejnowskiNeuronState
    vs::Float64
    va::Float64
    m::Float64
    h::Float64
    n::Float64
    kappa::Float64
    g_na::Float64
    g_k::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    c_s::Float64
    c_a::Float64
    dt::Float64
    v_threshold::Float64
end

function MainenSejnowskiNeuronState()
    MainenSejnowskiNeuronState(
        -65.0, -65.0, 0.05, 0.6, 0.3,
        10.0, 3000.0, 1500.0, 1.0,
        50.0, -90.0, -70.0,
        1.0, 0.1, 0.005, -20.0,
    )
end

"""Return whether every state and configuration field is finite and inside the public bounds."""
function valid(s::MainenSejnowskiNeuronState)
    values = (
        s.vs, s.va, s.m, s.h, s.n, s.kappa, s.g_na, s.g_k, s.g_l,
        s.e_na, s.e_k, s.e_l, s.c_s, s.c_a, s.dt, s.v_threshold,
    )
    all(isfinite, values) &&
        -200.0 <= s.vs <= 200.0 && -200.0 <= s.va <= 200.0 &&
        all(gate -> 0.0 <= gate <= 1.0, (s.m, s.h, s.n)) &&
        0.0 <= s.kappa <= 100.0 &&
        0.0 <= s.g_na <= 5000.0 && 0.0 <= s.g_k <= 3000.0 &&
        0.0 <= s.g_l <= 5.0 &&
        30.0 <= s.e_na <= 70.0 && -100.0 <= s.e_k <= -70.0 &&
        -90.0 <= s.e_l <= -50.0 &&
        0.5 <= s.c_s <= 2.0 && 0.05 <= s.c_a <= 1.0 &&
        0.0 < s.dt <= 0.1 && -40.0 <= s.v_threshold <= 20.0
end

function linoid(x::Float64, k::Float64)
    x == 0.0 ? k : x / -expm1(-x / k)
end

"""
    step!(state, current; dt=state.dt) -> Int

Advance the Mainen-Sejnowski two-compartment reduction by one discrete step and return the spike
indicator. Throws `ArgumentError` — with the pre-step state preserved
exactly — for a non-finite drive, an out-of-bounds configuration, a
`dt` that does not match the configured step, or a non-finite
candidate. State (vs, va, m, h, n) is committed only on success.
"""
function step!(s::MainenSejnowskiNeuronState, current::Float64=0.0; dt::Float64=s.dt)
    isfinite(current) || throw(ArgumentError("current must be finite"))
    valid(s) || throw(
        ArgumentError("Mainen-Sejnowski state and parameters must satisfy the public bounds")
    )
    dt == s.dt || throw(ArgumentError("dt must match the configured discrete step"))

    vs = s.vs
    va = s.va
    m = s.m
    h = s.h
    n = s.n
    v_prev = vs
    for _ in 1:20
        am = 0.182 * linoid(va + 25.0, 9.0)
        bm = 0.124 * linoid(-(va + 25.0), 9.0)
        ah = 0.024 * linoid(va + 40.0, 5.0)
        bh = 0.0091 * linoid(-(va + 65.0), 5.0)
        an = 0.02 * linoid(va - 20.0, 9.0)
        bn = 0.002 * linoid(-(va - 20.0), 9.0)

        m = clamp(m + (am * (1.0 - m) - bm * m) * s.dt, 0.0, 1.0)
        h = clamp(h + (ah * (1.0 - h) - bh * h) * s.dt, 0.0, 1.0)
        n = clamp(n + (an * (1.0 - n) - bn * n) * s.dt, 0.0, 1.0)

        i_na = s.g_na * m ^ 3 * h * (va - s.e_na)
        i_k = s.g_k * n * (va - s.e_k)
        i_l = s.g_l * (vs - s.e_l)

        dvs = (-i_l + s.kappa * (va - vs) + current) / s.c_s * s.dt
        dva = (-i_na - i_k + s.kappa * (vs - va)) / s.c_a * s.dt
        vs = clamp(vs + dvs, -200.0, 200.0)
        va = clamp(va + dva, -200.0, 200.0)

        all(isfinite, (vs, va, m, h, n)) ||
            throw(ArgumentError("Mainen-Sejnowski candidate state became non-finite"))
    end

    s.vs = vs
    s.va = va
    s.m = m
    s.h = h
    s.n = n
    (s.vs >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

"""Restore dynamic state to the initial values, preserving configuration."""
function reset!(s::MainenSejnowskiNeuronState)
    s.vs, s.va = -65.0, -65.0
    s.m, s.h, s.n = 0.05, 0.6, 0.3
    nothing
end

"""Run a fresh default-configured state for `n_steps` and return `(trace, spikes)`."""
function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.005)
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative"))
    s = MainenSejnowskiNeuronState()
    dt == s.dt || throw(ArgumentError("dt must match the configured discrete step"))
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        spikes += step!(s, I_ext; dt=dt)
        trace[t] = s.vs
    end
    trace, spikes
end

end # module MainenSejnowskiAccel
