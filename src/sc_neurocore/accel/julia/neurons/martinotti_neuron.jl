# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia RK4 kernel for the Martinotti adapting interneuron

module MartinottiNeuronAccel

export step!, simulate, MartinottiNeuronState

mutable struct MartinottiNeuronState
    v::Float64
    m::Float64
    h::Float64
    n::Float64
    p::Float64
    s::Float64
    g_na::Float64
    g_k::Float64
    g_m::Float64
    g_t::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_ca::Float64
    e_l::Float64
    c_m::Float64
    dt::Float64
    v_threshold::Float64
end

function MartinottiNeuronState()
    MartinottiNeuronState(-65.0, 0.02, 0.8, 0.2, 0.0, 0.9, 40.0, 5.0, 0.25, 0.01, 0.05, 50.0, -90.0, 120.0, -65.0, 0.8, 0.025, -20.0)
end

# L'Hôpital limit of the Traub-Miles x/(exp(x/slope)-1) rate at the singularity.
@inline function asing(num::Float64, slope::Float64, limit::Float64)
    return (abs(num) < 1e-6) ? limit : num / (exp(num / slope) - 1.0)
end

# Return (dV, dm, dh, dn, dp, ds) of the six-state system at one state.
@inline function derivatives(st::MartinottiNeuronState, v, m, h, n, p, s, current)
    dvt = v - (-56.2)
    alpha_m = -0.32 * asing(dvt - 13.0, -4.0, -4.0)
    beta_m = 0.28 * asing(dvt - 40.0, 5.0, 5.0)
    alpha_h = 0.128 * exp(-(dvt - 17.0) / 18.0)
    beta_h = 4.0 / (1.0 + exp(-(dvt - 40.0) / 5.0))
    alpha_n = -0.032 * asing(dvt - 15.0, -5.0, -5.0)
    beta_n = 0.5 * exp(-(dvt - 10.0) / 40.0)
    dm = alpha_m * (1.0 - m) - beta_m * m
    dh = alpha_h * (1.0 - h) - beta_h * h
    dn = alpha_n * (1.0 - n) - beta_n * n
    p_inf = 1.0 / (1.0 + exp(-(v + 35.0) / 10.0))
    tau_p = 400.0 / (3.3 * exp((v + 35.0) / 20.0) + exp(-(v + 35.0) / 20.0))
    dp = (p_inf - p) / tau_p
    m_t_inf = 1.0 / (1.0 + exp(-(v + 57.0) / 6.2))
    s_inf = 1.0 / (1.0 + exp((v + 81.0) / 4.0))
    tau_s = 30.0 + 200.0 / (1.0 + exp((v + 70.0) / 5.0))
    ds = (s_inf - s) / tau_s
    i_na = st.g_na * m * m * m * h * (v - st.e_na)
    i_k = st.g_k * n * n * n * n * (v - st.e_k)
    i_m = st.g_m * p * (v - st.e_k)
    i_t = st.g_t * m_t_inf * m_t_inf * s * (v - st.e_ca)
    i_l = st.g_l * (v - st.e_l)
    dv = (-i_na - i_k - i_m - i_t - i_l + current) / st.c_m
    return (dv, dm, dh, dn, dp, ds)
end

# One classical RK4 increment of (V, m, h, n, p, s), `current` held constant.
@inline function rk4_substep(st::MartinottiNeuronState, y::NTuple{6,Float64}, current::Float64)
    dt = st.dt
    k1 = derivatives(st, y[1], y[2], y[3], y[4], y[5], y[6], current)
    a = ntuple(i -> y[i] + 0.5 * dt * k1[i], 6)
    k2 = derivatives(st, a[1], a[2], a[3], a[4], a[5], a[6], current)
    b = ntuple(i -> y[i] + 0.5 * dt * k2[i], 6)
    k3 = derivatives(st, b[1], b[2], b[3], b[4], b[5], b[6], current)
    c = ntuple(i -> y[i] + dt * k3[i], 6)
    k4 = derivatives(st, c[1], c[2], c[3], c[4], c[5], c[6], current)
    return ntuple(i -> y[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0, 6)
end

function step!(s::MartinottiNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    v_prev = s.v
    y = (s.v, s.m, s.h, s.n, s.p, s.s)
    for _ in 1:4
        y = rk4_substep(s, y, I_ext)
    end
    s.v, s.m, s.h, s.n, s.p, s.s = y
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = MartinottiNeuronState()
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

end # module MartinottiNeuronAccel
