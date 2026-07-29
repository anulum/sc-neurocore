# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia mirror for IhNeuron

module IhNeuronAccel

export step!, simulate, reset!, valid, IhNeuronState

mutable struct IhNeuronState
    v::Float64
    h::Float64
    n::Float64
    r::Float64
    g_na::Float64
    g_k::Float64
    g_h::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_h::Float64
    e_l::Float64
    c_m::Float64
    phi::Float64
    dt::Float64
    v_threshold::Float64
    gain::Float64
    sub_steps::Int
end

function IhNeuronState()
    IhNeuronState(
        -65.0, 0.6, 0.32, 0.1,
        35.0, 9.0, 0.15, 0.2,
        55.0, -90.0, -40.0, -65.0,
        1.0, 5.0, 0.5, -20.0, 1.0, 50,
    )
end

function valid(s::IhNeuronState)
    values = (
        s.v, s.h, s.n, s.r, s.g_na, s.g_k, s.g_h, s.g_l,
        s.e_na, s.e_k, s.e_h, s.e_l, s.c_m, s.phi, s.dt,
        s.v_threshold, s.gain,
    )
    all(isfinite, values) &&
        -100.0 <= s.v <= 60.0 &&
        all(gate -> 0.0 <= gate <= 1.0, (s.h, s.n, s.r)) &&
        0.0 <= s.g_na <= 200.0 && 0.0 <= s.g_k <= 100.0 &&
        0.0 <= s.g_h <= 5.0 && 0.0 <= s.g_l <= 5.0 &&
        30.0 <= s.e_na <= 70.0 && -100.0 <= s.e_k <= -70.0 &&
        -50.0 <= s.e_h <= 0.0 && -80.0 <= s.e_l <= -40.0 &&
        0.5 <= s.c_m <= 2.0 && 0.5 <= s.phi <= 10.0 &&
        0.0 < s.dt <= 1.0 && -20.0 <= s.v_threshold <= 20.0 &&
        0.0 <= s.gain <= 10.0 && 1 <= s.sub_steps <= 10_000
end

function safe_rate(a::Float64, vhalf::Float64, v::Float64, k::Float64, fallback::Float64)
    d = v + vhalf
    abs(d) < 1.0e-7 ? fallback : a * d / (1.0 - exp(-d / k))
end

function step!(s::IhNeuronState, current::Float64=0.0; dt::Float64=s.dt)
    isfinite(current) || throw(ArgumentError("current must be finite"))
    valid(s) || throw(ArgumentError("Ih state and parameters must satisfy the public bounds"))
    dt == s.dt || throw(ArgumentError("dt must match the configured discrete step"))

    v = s.v
    h = s.h
    n = s.n
    r = s.r
    input = s.gain * current
    sub_dt = s.dt / s.sub_steps
    fired = 0
    for _ in 1:s.sub_steps
        alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0)
        beta_m = 4.0 * exp(-(v + 60.0) / 18.0)
        m_inf = alpha_m / (alpha_m + beta_m)
        alpha_h = 0.07 * exp(-(v + 58.0) / 20.0)
        beta_h = 1.0 / (1.0 + exp(-(v + 28.0) / 10.0))
        alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1)
        beta_n = 0.125 * exp(-(v + 44.0) / 80.0)
        r_inf = 1.0 / (1.0 + exp((v + 80.0) / 10.0))
        tau_r = 100.0 + 200.0 / (1.0 + exp((v + 70.0) / 10.0))

        h += sub_dt * s.phi * (alpha_h * (1.0 - h) - beta_h * h)
        n += sub_dt * s.phi * (alpha_n * (1.0 - n) - beta_n * n)
        r += sub_dt * (r_inf - r) / tau_r
        i_na = s.g_na * m_inf ^ 3 * h * (v - s.e_na)
        i_k = s.g_k * n ^ 4 * (v - s.e_k)
        i_h = s.g_h * r * (v - s.e_h)
        i_l = s.g_l * (v - s.e_l)
        v += sub_dt * (-i_na - i_k - i_h - i_l + input) / s.c_m
        all(isfinite, (v, h, n, r)) ||
            throw(ArgumentError("Ih candidate state became non-finite"))
        if v >= s.v_threshold
            fired = 1
            v = -65.0
        end
    end

    s.v = clamp(v, -100.0, 60.0)
    s.h = clamp(h, 0.0, 1.0)
    s.n = clamp(n, 0.0, 1.0)
    s.r = clamp(r, 0.0, 1.0)
    fired
end

function reset!(s::IhNeuronState)
    s.v, s.h, s.n, s.r = -65.0, 0.6, 0.32, 0.1
    nothing
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.5)
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative"))
    s = IhNeuronState()
    dt == s.dt || throw(ArgumentError("dt must match the configured discrete step"))
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        spikes += step!(s, I_ext; dt=dt)
        trace[t] = s.v
    end
    trace, spikes
end

end # module IhNeuronAccel
