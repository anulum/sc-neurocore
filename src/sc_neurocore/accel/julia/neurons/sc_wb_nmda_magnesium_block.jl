# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia retained SC WB plus NMDA recurrence

module SCWBNMDAMagnesiumBlockAccel
export step!, reset!, valid, SCWBNMDAMagnesiumBlockNeuronState

mutable struct SCWBNMDAMagnesiumBlockNeuronState
    v::Float64; h::Float64; n::Float64; s_nmda::Float64; g_na::Float64; g_k::Float64
    g_nmda::Float64; g_l::Float64; e_na::Float64; e_k::Float64; e_nmda::Float64; e_l::Float64
    c_m::Float64; phi::Float64; mg_conc::Float64; tau_rise::Float64; tau_decay::Float64
    dt::Float64; v_threshold::Float64; gain::Float64; sub_steps::Int
end
SCWBNMDAMagnesiumBlockNeuronState() = SCWBNMDAMagnesiumBlockNeuronState(
    -65.0, 0.6, 0.32, 0.0, 35.0, 9.0, 0.5, 0.1, 55.0, -90.0, 0.0, -65.0,
    1.0, 5.0, 1.0, 10.0, 100.0, 0.5, -20.0, 1.0, 50,
)

safe_rate(a, vhalf, v, k, fallback) =
    abs(v + vhalf) < 1e-7 ? fallback : a * (v + vhalf) / (1 - exp(-(v + vhalf) / k))

function valid(s::SCWBNMDAMagnesiumBlockNeuronState)
    values = (
        s.v, s.h, s.n, s.s_nmda, s.g_na, s.g_k, s.g_nmda, s.g_l, s.e_na,
        s.e_k, s.e_nmda, s.e_l, s.c_m, s.phi, s.mg_conc, s.tau_rise,
        s.tau_decay, s.dt, s.v_threshold, s.gain,
    )
    all(isfinite, values) && -100 <= s.v <= 60 &&
        all(x -> 0 <= x <= 1, (s.h, s.n, s.s_nmda)) && 0 <= s.g_na <= 200 &&
        0 <= s.g_k <= 100 && 0 <= s.g_nmda <= 20 && 0 <= s.g_l <= 5 &&
        30 <= s.e_na <= 70 && -100 <= s.e_k <= -70 && -10 <= s.e_nmda <= 10 &&
        -80 <= s.e_l <= -40 && 0.5 <= s.c_m <= 2 && 0.5 <= s.phi <= 10 &&
        0 <= s.mg_conc <= 5 && 0.1 <= s.tau_rise <= 20 &&
        10 <= s.tau_decay <= 500 && 0 < s.dt <= 1 &&
        -20 <= s.v_threshold <= 20 && 0 <= s.gain <= 10 &&
        1 <= s.sub_steps <= 10000
end

function step!(s::SCWBNMDAMagnesiumBlockNeuronState, current::Float64=0.0)
    isfinite(current) || throw(ArgumentError("current must be finite"))
    valid(s) || throw(ArgumentError("SC NMDA state and parameters must satisfy the public bounds"))
    v = s.v
    h = s.h
    n = s.n
    input = s.gain * current
    sub_dt = s.dt / s.sub_steps
    drive = input > 0 ? input / (input + 5) : 0.0
    tau = drive > s.s_nmda ? s.tau_rise : s.tau_decay
    g = clamp(s.s_nmda + s.dt * (drive - s.s_nmda) / tau, 0, 1)
    event = 0
    for _ in 1:s.sub_steps
        am = safe_rate(0.1, 35.0, v, 10.0, 1.0)
        bm = 4 * exp(-(v + 60) / 18)
        mi = am / (am + bm)
        ah = 0.07 * exp(-(v + 58) / 20)
        bh = 1 / (1 + exp(-(v + 28) / 10))
        an = safe_rate(0.01, 34.0, v, 10.0, 0.1)
        bn = 0.125 * exp(-(v + 44) / 80)
        block = 1 / (1 + (s.mg_conc / 3.57) * exp(-0.062 * v))
        h += sub_dt * s.phi * (ah * (1 - h) - bh * h)
        n += sub_dt * s.phi * (an * (1 - n) - bn * n)
        ina = s.g_na * mi^3 * h * (v - s.e_na)
        ik = s.g_k * n^4 * (v - s.e_k)
        inmda = s.g_nmda * g * block * (v - s.e_nmda)
        il = s.g_l * (v - s.e_l)
        v += sub_dt * (-ina - ik - inmda - il + input) / s.c_m
        all(isfinite, (v, h, n)) ||
            throw(ArgumentError("SC NMDA candidate state became non-finite"))
        if v >= s.v_threshold
            event = 1
            v = -65.0
        end
    end
    s.v = clamp(v, -100, 60)
    s.h = clamp(h, 0, 1)
    s.n = clamp(n, 0, 1)
    s.s_nmda = g
    event
end

function reset!(s::SCWBNMDAMagnesiumBlockNeuronState)
    s.v = -65.0
    s.h = 0.6
    s.n = 0.32
    s.s_nmda = 0.0
    nothing
end
end # module SCWBNMDAMagnesiumBlockAccel
