# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Hill-Tononi 2005 hybrid neuron

module HillTononiAccel

export step!, reset!, simulate, HillTononiNeuronState

"""Hill-Tononi cortical-waking state and complete scalar-cell configuration."""
mutable struct HillTononiNeuronState
    v::Float64
    theta::Float64
    d_k::Float64
    m_h::Float64
    m_t::Float64
    h_t::Float64
    spike_timer::Float64
    g_na_l::Float64
    g_k_l::Float64
    g_na_p::Float64
    g_dk::Float64
    g_h::Float64
    g_t::Float64
    e_na::Float64
    e_k::Float64
    e_na_p::Float64
    e_dk::Float64
    e_h::Float64
    e_t::Float64
    n_na_p::Float64
    n_t::Float64
    tau_m::Float64
    theta_eq::Float64
    tau_theta::Float64
    g_spike::Float64
    t_spike::Float64
    tau_spike::Float64
    tau_d::Float64
    d_influx_peak::Float64
    d_threshold::Float64
    d_slope::Float64
    d_eq::Float64
    d_half::Float64
    dt::Float64
end

"""Return the publication's cortical-excitatory waking profile."""
function HillTononiNeuronState()
    HillTononiNeuronState(
        -70.0, -51.0, 0.001,
        0.2871859013825026, 0.1450215950687922, 0.03732688734412946, 0.0,
        0.2, 1.0, 0.5, 0.5, 0.0, 0.0,
        30.0, -90.0, 30.0, -90.0, -40.0, 0.0, 3.0, 2.0,
        16.0, -51.0, 2.0, 1.0, 2.0, 1.75,
        1250.0, 0.025, -10.0, 5.0, 0.001, 0.25, 0.25,
    )
end

_m_h_inf(v) = 1.0 / (1.0 + exp((v + 75.0) / 5.5))
_tau_m_h(v) = 1.0 / (exp(-14.59 - 0.086 * v) + exp(-1.87 + 0.0701 * v))
_m_t_inf(v) = 1.0 / (1.0 + exp(-(v + 59.0) / 6.2))
_tau_m_t(v) = 0.22 / (exp(-(v + 132.0) / 16.7) + exp((v + 16.8) / 18.2)) + 0.13
_h_t_inf(v) = 1.0 / (1.0 + exp((v + 83.0) / 4.0))
_tau_h_t(v) = 8.2 + (56.6 + 0.27 * exp((v + 115.2) / 5.0)) / (1.0 + exp((v + 86.0) / 3.2))

function _d_k_inf(s::HillTononiNeuronState, v::Float64)
    influx = s.d_influx_peak / (1.0 + exp(-(v - s.d_threshold) / s.d_slope))
    return s.tau_d * influx + s.d_eq
end

function _derivatives(s::HillTononiNeuronState, y::NTuple{6,Float64}, current::Float64, spike_active::Bool)
    v, theta, d_k, m_h, m_t, h_t = y
    m_na_p = 1.0 / (1.0 + exp(-(v + 55.7) / 7.7))
    d_activation = 1.0 / (1.0 + (s.d_half / max(d_k, 1e-15))^3.5)
    i_na_l = -s.g_na_l * (v - s.e_na)
    i_k_l = -s.g_k_l * (v - s.e_k)
    i_na_p = -s.g_na_p * m_na_p^s.n_na_p * (v - s.e_na_p)
    i_dk = -s.g_dk * d_activation * (v - s.e_dk)
    i_h = -s.g_h * m_h * (v - s.e_h)
    i_t = -s.g_t * m_t^s.n_t * h_t * (v - s.e_t)
    i_spike = spike_active ? -s.g_spike * (v - s.e_k) / s.tau_spike : 0.0
    return (
        (i_na_l + i_k_l + i_na_p + i_dk + i_h + i_t + current) / s.tau_m + i_spike,
        -(theta - s.theta_eq) / s.tau_theta,
        (_d_k_inf(s, v) - d_k) / s.tau_d,
        (_m_h_inf(v) - m_h) / _tau_m_h(v),
        (_m_t_inf(v) - m_t) / _tau_m_t(v),
        (_h_t_inf(v) - h_t) / _tau_h_t(v),
    )
end

function _rk4_candidate(s::HillTononiNeuronState, y::NTuple{6,Float64}, current::Float64, spike_active::Bool)
    dt = s.dt
    k1 = _derivatives(s, y, current, spike_active)
    y2 = ntuple(i -> y[i] + 0.5 * dt * k1[i], 6)
    k2 = _derivatives(s, y2, current, spike_active)
    y3 = ntuple(i -> y[i] + 0.5 * dt * k2[i], 6)
    k3 = _derivatives(s, y3, current, spike_active)
    y4 = ntuple(i -> y[i] + dt * k3[i], 6)
    k4 = _derivatives(s, y4, current, spike_active)
    return ntuple(i -> y[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0, 6)
end

function _configuration_valid(s::HillTononiNeuronState)
    values = (
        s.v, s.theta, s.d_k, s.m_h, s.m_t, s.h_t, s.spike_timer,
        s.g_na_l, s.g_k_l, s.g_na_p, s.g_dk, s.g_h, s.g_t,
        s.e_na, s.e_k, s.e_na_p, s.e_dk, s.e_h, s.e_t, s.n_na_p, s.n_t,
        s.tau_m, s.theta_eq, s.tau_theta, s.g_spike, s.t_spike, s.tau_spike,
        s.tau_d, s.d_influx_peak, s.d_threshold, s.d_slope, s.d_eq, s.d_half, s.dt,
    )
    nonnegative = (
        s.g_na_l, s.g_k_l, s.g_na_p, s.g_dk, s.g_h, s.g_t,
        s.g_spike, s.d_influx_peak, s.d_eq,
    )
    positive = (
        s.n_na_p, s.n_t, s.tau_m, s.tau_theta, s.t_spike,
        s.tau_spike, s.tau_d, s.d_slope, s.d_half, s.dt,
    )
    return all(isfinite, values) && s.d_k >= 0.0 && s.spike_timer >= 0.0 &&
           all(value -> value >= 0.0, nonnegative) && all(value -> value > 0.0, positive)
end

"""Advance one source RK4 step and commit state only after validation."""
function step!(s::HillTononiNeuronState, current::Float64=0.0)
    isfinite(current) && _configuration_valid(s) ||
        throw(DomainError(current, "Hill-Tononi state and current must be finite and physical"))
    refractory = s.spike_timer > 0.0
    state = (s.v, s.theta, s.d_k, s.m_h, s.m_t, s.h_t)
    candidate = _rk4_candidate(s, state, current, refractory)
    all(isfinite, candidate) && candidate[3] >= 0.0 ||
        throw(DomainError(candidate, "Hill-Tononi candidate must be finite and physical"))
    timer = max(0.0, s.spike_timer - s.dt)
    spike = !refractory && candidate[1] >= candidate[2]
    if spike
        candidate = (s.e_na, s.e_na, candidate[3], candidate[4], candidate[5], candidate[6])
        timer = s.t_spike
    end
    s.v, s.theta, s.d_k, s.m_h, s.m_t, s.h_t = candidate
    s.spike_timer = timer
    return spike ? 1 : 0
end

"""Restore the source cortical-excitatory waking initial state."""
function reset!(s::HillTononiNeuronState)
    s.v, s.theta, s.d_k = -70.0, -51.0, 0.001
    s.m_h, s.m_t, s.h_t = _m_h_inf(s.v), _m_t_inf(s.v), _h_t_inf(s.v)
    s.spike_timer = 0.0
    return s
end

"""Simulate `n_steps` at constant external current and return voltage and events."""
function simulate(n_steps::Int=1000; I_ext::Float64=20.0)
    state = HillTononiNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for index in 1:n_steps
        spikes += step!(state, I_ext)
        trace[index] = state.v
    end
    return trace, spikes
end

end # module HillTononiAccel
