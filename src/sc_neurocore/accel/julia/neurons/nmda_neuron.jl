# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Wang 1999 NMDA-autapse neuron

module NmdaNeuronAccel

export step!, simulate, reset!, valid, NMDANeuronState

mutable struct NMDANeuronState
    v::Float64; x_nmda::Float64; s_nmda::Float64; ca::Float64; refractory_remaining::Float64
    c_m::Float64; g_l::Float64; v_l::Float64; g_nmda::Float64; e_nmda::Float64; mg_conc::Float64
    alpha_x::Float64; tau_x::Float64; alpha_s::Float64; tau_s::Float64; kinetic_scale::Float64
    g_ahp::Float64; v_k::Float64; alpha_ca::Float64; tau_ca::Float64
    dt::Float64; v_threshold::Float64; v_reset::Float64; refractory_period::Float64
end

NMDANeuronState() = NMDANeuronState(
    -70.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.025, -70.0, 0.1, 0.0, 1.0,
    1.0, 2.0, 1.0, 80.0, 1.0, 0.0, -85.0, 0.2, 80.0, 0.05, -52.0, -59.0, 2.0,
)

function valid(s::NMDANeuronState)
    values = (s.v,s.x_nmda,s.s_nmda,s.ca,s.refractory_remaining,s.c_m,s.g_l,s.v_l,
        s.g_nmda,s.e_nmda,s.mg_conc,s.alpha_x,s.tau_x,s.alpha_s,s.tau_s,s.kinetic_scale,
        s.g_ahp,s.v_k,s.alpha_ca,s.tau_ca,s.dt,s.v_threshold,s.v_reset,s.refractory_period)
    all(isfinite, values) && -120.0 <= s.v <= 80.0 && s.x_nmda >= 0.0 &&
        0.0 <= s.s_nmda <= 1.0 && s.ca >= 0.0 &&
        0.0 <= s.refractory_remaining <= s.refractory_period &&
        0.01 <= s.c_m <= 10.0 && 0.0 <= s.g_l <= 1.0 && -100.0 <= s.v_l <= -40.0 &&
        0.0 <= s.g_nmda <= 2.0 && -10.0 <= s.e_nmda <= 10.0 && 0.0 <= s.mg_conc <= 5.0 &&
        0.0 <= s.alpha_x <= 10.0 && 0.01 <= s.tau_x <= 100.0 &&
        0.0 <= s.alpha_s <= 10.0 && 1.0 <= s.tau_s <= 1000.0 &&
        0.01 <= s.kinetic_scale <= 100.0 && 0.0 <= s.g_ahp <= 10.0 &&
        -120.0 <= s.v_k <= -40.0 && 0.0 <= s.alpha_ca <= 10.0 &&
        1.0 <= s.tau_ca <= 1000.0 && 0.0 < s.dt <= 0.05 &&
        -80.0 <= s.v_threshold <= -30.0 && -100.0 <= s.v_reset < s.v_threshold &&
        0.0 <= s.refractory_period <= 20.0
end

function derivatives(s::NMDANeuronState, v, x, g, ca, current)
    block = 1.0 / (1.0 + s.mg_conc * exp(-0.062 * v) / 3.57)
    i_l = s.g_l * (v - s.v_l); i_ahp = s.g_ahp * ca * (v - s.v_k)
    i_nmda = s.g_nmda * g * block * (v - s.e_nmda)
    ((-i_l - i_ahp - i_nmda + current) / s.c_m,
        s.kinetic_scale * (-x / s.tau_x),
        s.kinetic_scale * (s.alpha_s * x * (1.0 - g) - g / s.tau_s), -ca / s.tau_ca)
end

function rk2(s::NMDANeuronState, v, current)
    y = (v, s.x_nmda, s.s_nmda, s.ca); k1 = derivatives(s, y..., current); h = 0.5 * s.dt
    midpoint = ntuple(i -> y[i] + h * k1[i], 4); k2 = derivatives(s, midpoint..., current)
    ntuple(i -> y[i] + s.dt * k2[i], 4)
end

"""Advance one Wang source-grid step atomically."""
function step!(s::NMDANeuronState, current::Float64=0.0; dt::Float64=s.dt)
    isfinite(current) || throw(ArgumentError("current must be finite"))
    valid(s) || throw(ArgumentError("NMDA state and parameters must satisfy the public bounds"))
    dt == s.dt || throw(ArgumentError("dt must match the configured discrete step"))
    held = s.refractory_remaining > 0.0
    y = collect(rk2(s, held ? s.v_reset : s.v, current))
    refractory = max(0.0, s.refractory_remaining - s.dt); event = 0
    if held
        y[1] = s.v_reset
    elseif y[1] >= s.v_threshold
        event = 1; y[1] = s.v_reset; refractory = s.refractory_period
        y[2] += s.kinetic_scale * s.alpha_x; y[4] += s.alpha_ca
    end
    all(isfinite, (y..., refractory)) || throw(ArgumentError("NMDA candidate state became non-finite"))
    s.v = clamp(y[1], -120.0, 80.0); s.x_nmda = max(0.0, y[2]); s.s_nmda = clamp(y[3], 0.0, 1.0)
    s.ca = max(0.0, y[4]); s.refractory_remaining = refractory; event
end

function reset!(s::NMDANeuronState)
    s.v = s.v_l; s.x_nmda = 0.0; s.s_nmda = 0.0; s.ca = 0.0; s.refractory_remaining = 0.0; nothing
end

function simulate(n_steps::Int=1000; I_ext::Float64=0.6, dt::Float64=0.05)
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative")); s=NMDANeuronState()
    dt == s.dt || throw(ArgumentError("dt must match the configured discrete step"))
    trace=zeros(n_steps); events=0
    for i in eachindex(trace); events += step!(s,I_ext); trace[i]=s.v; end
    trace,events
end

end # module NmdaNeuronAccel
