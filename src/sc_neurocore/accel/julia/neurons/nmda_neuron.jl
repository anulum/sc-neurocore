# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia mirror for NMDANeuron

module NmdaNeuronAccel

export step!, simulate, reset!, valid, NMDANeuronState

mutable struct NMDANeuronState
    v::Float64
    h::Float64
    n::Float64
    s_nmda::Float64
    g_na::Float64
    g_k::Float64
    g_nmda::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_nmda::Float64
    e_l::Float64
    c_m::Float64
    phi::Float64
    mg_conc::Float64
    tau_rise::Float64
    tau_decay::Float64
    dt::Float64
    v_threshold::Float64
    gain::Float64
    sub_steps::Int
end

function NMDANeuronState()
    NMDANeuronState(
        -65.0, 0.6, 0.32, 0.0,
        35.0, 9.0, 0.5, 0.1,
        55.0, -90.0, 0.0, -65.0,
        1.0, 5.0, 1.0, 10.0, 100.0,
        0.5, -20.0, 1.0, 50,
    )
end

function valid(s::NMDANeuronState)
    values = (
        s.v, s.h, s.n, s.s_nmda, s.g_na, s.g_k, s.g_nmda, s.g_l,
        s.e_na, s.e_k, s.e_nmda, s.e_l, s.c_m, s.phi, s.mg_conc,
        s.tau_rise, s.tau_decay, s.dt, s.v_threshold, s.gain,
    )
    all(isfinite, values) &&
        -100.0 <= s.v <= 60.0 &&
        all(gate -> 0.0 <= gate <= 1.0, (s.h, s.n, s.s_nmda)) &&
        0.0 <= s.g_na <= 200.0 && 0.0 <= s.g_k <= 100.0 &&
        0.0 <= s.g_nmda <= 20.0 && 0.0 <= s.g_l <= 5.0 &&
        30.0 <= s.e_na <= 70.0 && -100.0 <= s.e_k <= -70.0 &&
        -10.0 <= s.e_nmda <= 10.0 && -80.0 <= s.e_l <= -40.0 &&
        0.5 <= s.c_m <= 2.0 && 0.5 <= s.phi <= 10.0 &&
        0.0 <= s.mg_conc <= 5.0 &&
        0.1 <= s.tau_rise <= 20.0 && 10.0 <= s.tau_decay <= 500.0 &&
        0.0 < s.dt <= 1.0 && -20.0 <= s.v_threshold <= 20.0 &&
        0.0 <= s.gain <= 10.0 && 1 <= s.sub_steps <= 10_000
end

function safe_rate(a::Float64, vhalf::Float64, v::Float64, k::Float64, fallback::Float64)
    d = v + vhalf
    abs(d) < 1.0e-7 ? fallback : a * d / (1.0 - exp(-d / k))
end

function step!(s::NMDANeuronState, current::Float64=0.0; dt::Float64=s.dt)
    isfinite(current) || throw(ArgumentError("current must be finite"))
    valid(s) || throw(ArgumentError("NMDA state and parameters must satisfy the public bounds"))
    dt == s.dt || throw(ArgumentError("dt must match the configured discrete step"))

    v = s.v
    h = s.h
    n = s.n
    input = s.gain * current
    sub_dt = s.dt / s.sub_steps
    fired = 0

    drive = (input > 0.0) ? input / (input + 5.0) : 0.0
    tau = (drive > s.s_nmda) ? s.tau_rise : s.tau_decay
    ds = (drive - s.s_nmda) / tau
    s_candidate = s.s_nmda + s.dt * ds
    s_candidate = max(0.0, min(1.0, s_candidate))

    for _ in 1:s.sub_steps
        alpha_m = safe_rate(0.1, 35.0, v, 10.0, 1.0)
        beta_m = 4.0 * exp(-(v + 60.0) / 18.0)
        m_inf = alpha_m / (alpha_m + beta_m)
        alpha_h = 0.07 * exp(-(v + 58.0) / 20.0)
        beta_h = 1.0 / (1.0 + exp(-(v + 28.0) / 10.0))
        alpha_n = safe_rate(0.01, 34.0, v, 10.0, 0.1)
        beta_n = 0.125 * exp(-(v + 44.0) / 80.0)
        mg_block = 1.0 / (1.0 + (s.mg_conc / 3.57) * exp(-0.062 * v))

        h += sub_dt * s.phi * (alpha_h * (1.0 - h) - beta_h * h)
        n += sub_dt * s.phi * (alpha_n * (1.0 - n) - beta_n * n)
        i_na = s.g_na * m_inf ^ 3 * h * (v - s.e_na)
        i_k = s.g_k * n ^ 4 * (v - s.e_k)
        i_nmda = s.g_nmda * s_candidate * mg_block * (v - s.e_nmda)
        i_l = s.g_l * (v - s.e_l)
        v += sub_dt * (-i_na - i_k - i_nmda - i_l + input) / s.c_m
        all(isfinite, (v, h, n)) ||
            throw(ArgumentError("NMDA candidate state became non-finite"))
        if v >= s.v_threshold
            fired = 1
            v = -65.0
        end
    end

    s.v = clamp(v, -100.0, 60.0)
    s.h = clamp(h, 0.0, 1.0)
    s.n = clamp(n, 0.0, 1.0)
    s.s_nmda = s_candidate
    fired
end

function reset!(s::NMDANeuronState)
    s.v, s.h, s.n, s.s_nmda = -65.0, 0.6, 0.32, 0.0
    nothing
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.5)
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative"))
    s = NMDANeuronState()
    dt == s.dt || throw(ArgumentError("dt must match the configured discrete step"))
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        spikes += step!(s, I_ext; dt=dt)
        trace[t] = s.v
    end
    trace, spikes
end

end # module NmdaNeuronAccel
