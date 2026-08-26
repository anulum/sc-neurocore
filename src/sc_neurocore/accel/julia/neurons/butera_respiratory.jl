# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for butera_respiratory

module ButeraRespiratoryAccel

export step!, simulate, validate_butera_respiratory, ButeraRespiratoryNeuronState

mutable struct ButeraRespiratoryNeuronState
    v::Float64
    n::Float64
    h_nap::Float64
    g_na::Float64
    g_nap::Float64
    g_k::Float64
    g_l::Float64
    capacitance::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    g_tonic::Float64
    e_syn::Float64
    tau_h::Float64
    dt::Float64
    v_threshold::Float64
end

function ButeraRespiratoryNeuronState()
    ButeraRespiratoryNeuronState(-50.0, 0.01, 0.5, 28.0, 2.8, 11.2, 2.8, 21.0, 50.0, -85.0, -65.0, 0.0, 0.0, 10000.0, 0.1, -20.0)
end

@inline _finite(xs...) = all(isfinite, xs)
@inline _state_valid(v, n, h) = _finite(v, n, h) && -200.0 <= v <= 100.0 && -0.05 <= n <= 1.05 && -0.05 <= h <= 1.05

function validate_butera_respiratory(s::ButeraRespiratoryNeuronState)::Bool
    return _finite(s.v, s.n, s.h_nap, s.g_na, s.g_nap, s.g_k, s.g_l, s.capacitance, s.e_na, s.e_k, s.e_l, s.g_tonic, s.e_syn, s.tau_h, s.dt, s.v_threshold) &&
        s.g_na >= 0.0 && s.g_nap >= 0.0 && s.g_k >= 0.0 && s.g_l >= 0.0 && s.capacitance > 0.0 && s.g_tonic >= 0.0 && s.tau_h > 0.0 && s.dt > 0.0 &&
        _state_valid(s.v, s.n, s.h_nap)
end

function _rates(v::Float64, tau_h_base::Float64)
    m_na = 1.0 / (1.0 + exp(-(v + 34.0) / 5.0))
    m_nap = 1.0 / (1.0 + exp(-(v + 40.0) / 6.0))
    h_inf = 1.0 / (1.0 + exp((v + 48.0) / 6.0))
    n_inf = 1.0 / (1.0 + exp(-(v + 29.0) / 4.0))
    tau_n = max(10.0 / max(cosh((v + 29.0) / 8.0), 1e-12), 0.01)
    tau_h = max(tau_h_base / max(cosh((v + 48.0) / 12.0), 1e-12), 0.1)
    rates = (m_na, m_nap, h_inf, n_inf, tau_n, tau_h)
    return all(isfinite, rates) ? rates : nothing
end

function _derivatives(s::ButeraRespiratoryNeuronState, state::NTuple{3,Float64}, I_ext::Float64)
    v, n, h_nap = state
    (!all(isfinite, state) || !isfinite(I_ext)) && return nothing
    v = clamp(v, -200.0, 100.0)
    n = clamp(n, 0.0, 1.0)
    h_nap = clamp(h_nap, 0.0, 1.0)
    rates = _rates(v, s.tau_h)
    rates === nothing && return nothing
    m_na, m_nap, h_inf, n_inf, tau_n, tau_h = rates
    i_na = s.g_na * m_na^3 * (1.0 - n) * (v - s.e_na)
    i_nap = s.g_nap * m_nap * h_nap * (v - s.e_na)
    i_k = s.g_k * n^4 * (v - s.e_k)
    i_l = s.g_l * (v - s.e_l)
    i_tonic = s.g_tonic * (v - s.e_syn)
    deriv = ((-i_na - i_nap - i_k - i_l - i_tonic + I_ext) / s.capacitance, (n_inf - n) / tau_n, (h_inf - h_nap) / tau_h)
    return all(isfinite, deriv) ? deriv : nothing
end

function _rk4_candidate(s::ButeraRespiratoryNeuronState, I_ext::Float64)
    (!validate_butera_respiratory(s) || !isfinite(I_ext)) && return nothing
    state = (s.v, s.n, s.h_nap)
    k1 = _derivatives(s, state, I_ext)
    k1 === nothing && return nothing
    k2 = _derivatives(s, ntuple(i -> state[i] + 0.5 * s.dt * k1[i], 3), I_ext)
    k2 === nothing && return nothing
    k3 = _derivatives(s, ntuple(i -> state[i] + 0.5 * s.dt * k2[i], 3), I_ext)
    k3 === nothing && return nothing
    k4 = _derivatives(s, ntuple(i -> state[i] + s.dt * k3[i], 3), I_ext)
    k4 === nothing && return nothing
    candidate = ntuple(i -> state[i] + s.dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0, 3)
    !all(isfinite, candidate) && return nothing
    return (clamp(candidate[1], -200.0, 100.0), clamp(candidate[2], 0.0, 1.0), clamp(candidate[3], 0.0, 1.0))
end

function step!(s::ButeraRespiratoryNeuronState, I_ext::Float64=0.0; dt::Union{Nothing,Float64}=nothing)
    if dt !== nothing
        (!isfinite(dt) || dt <= 0.0) && return -1
        s.dt = dt
    end
    v_prev = s.v
    candidate = _rk4_candidate(s, I_ext)
    candidate === nothing && return -1
    s.v, s.n, s.h_nap = candidate
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = ButeraRespiratoryNeuronState()
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

end # module ButeraRespiratoryAccel
