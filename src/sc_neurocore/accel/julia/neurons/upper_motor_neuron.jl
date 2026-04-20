# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for upper_motor_neuron

module UpperMotorNeuronAccel

export step!, simulate, UpperMotorNeuronState

mutable struct UpperMotorNeuronState
    v::Float64
    m::Float64
    h::Float64
    n::Float64
    p::Float64
    s::Float64
    g_na::Float64
    g_k::Float64
    g_m::Float64
    g_ca::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_ca::Float64
    e_l::Float64
    c_m::Float64
    dt::Float64
    v_threshold::Float64
end

function UpperMotorNeuronState()
    UpperMotorNeuronState(-70.0, 0.05, 0.6, 0.3, 0.0, 0.0, 50.0, 5.0, 0.07, 0.3, 0.1, 50.0, -90.0, 120.0, -70.0, 1.0, 0.025, -20.0)
end

function step!(s::UpperMotorNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        vt = -56.2
        for _ in 1:4
            dv = s.v - vt
            x_m = dv - 13.0
            alpha_m = (abs(x_m) < 1e-06) ? 0.32 * 4.0 : -0.32 * x_m / (exp(-x_m / 4.0) - 1.0)
            x_h = dv - 17.0
            beta_m = (abs(x_h) < 1e-06) ? 0.28 * 5.0 : 0.28 * x_h / (exp(x_h / 5.0) - 1.0)
            alpha_h = 0.128 * exp(-(dv - 17.0) / 18.0)
            beta_h = 4.0 / (1.0 + exp(-(dv - 40.0) / 5.0))
            x_n = dv - 15.0
            alpha_n = (abs(x_n) < 1e-06) ? 0.032 * 5.0 : -0.032 * x_n / (exp(-x_n / 5.0) - 1.0)
            beta_n = 0.5 * exp(-(dv - 10.0) / 40.0)
            s.m += (alpha_m * (1.0 - s.m) - beta_m * s.m) * s.dt
            s.h += (alpha_h * (1.0 - s.h) - beta_h * s.h) * s.dt
            s.n += (alpha_n * (1.0 - s.n) - beta_n * s.n) * s.dt
            p_inf = 1.0 / (1.0 + exp(-(s.v + 35.0) / 10.0))
            tau_p = 400.0 / (3.3 * exp((s.v + 35.0) / 20.0) + exp(-(s.v + 35.0) / 20.0))
            s.p += (p_inf - s.p) / tau_p * s.dt
            s_inf = 1.0 / (1.0 + exp(-(s.v + 20.0) / 5.0))
            s.s += (s_inf - s.s) / 10.0 * s.dt
            i_na = s.g_na * s.m ^ 3 * s.h * (s.v - s.e_na)
            i_k = s.g_k * s.n ^ 4 * (s.v - s.e_k)
            i_m = s.g_m * s.p * (s.v - s.e_k)
            i_ca = s.g_ca * s.s ^ 2 * (s.v - s.e_ca)
            i_l = s.g_l * (s.v - s.e_l)
            s.v += (-i_na - i_k - i_m - i_ca - i_l + I_ext) / s.c_m * s.dt
        end
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = UpperMotorNeuronState()
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

end # module UpperMotorNeuronAccel
