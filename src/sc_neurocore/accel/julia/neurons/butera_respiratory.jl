# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for butera_respiratory

module ButeraRespiratoryAccel

export step!, simulate, ButeraRespiratoryNeuronState

mutable struct ButeraRespiratoryNeuronState
    v::Float64
    n::Float64
    h_nap::Float64
    g_na::Float64
    g_nap::Float64
    g_k::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    e_syn::Float64
    tau_h::Float64
    dt::Float64
    v_threshold::Float64
end

function ButeraRespiratoryNeuronState()
    ButeraRespiratoryNeuronState(-50.0, 0.01, 0.5, 28.0, 2.8, 11.2, 2.8, 50.0, -85.0, -65.0, -10.0, 10000.0, 0.1, -20.0)
end

function _sexp(s::ButeraRespiratoryNeuronState, x)
    return Float64(exp(clamp(x, -500, 500)))
end

function _scosh(s::ButeraRespiratoryNeuronState, x)
    cx = clamp(x, -500, 500)
    return Float64(cosh(cx))
end

function step!(s::ButeraRespiratoryNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        v_prev = s.v
        m_na_inf = 1.0 / (1.0 + s._sexp(-(s.v + 34.0) / 5.0))
        m_nap_inf = 1.0 / (1.0 + s._sexp(-(s.v + 40.0) / 6.0))
        h_nap_inf = 1.0 / (1.0 + s._sexp((s.v + 48.0) / 6.0))
        n_inf = 1.0 / (1.0 + s._sexp(-(s.v + 29.0) / 4.0))
        tau_n = 10.0 / max(s._scosh((s.v + 29.0) / 8.0), 1e-12)
        tau_h = s.tau_h / max(s._scosh((s.v + 48.0) / 12.0), 1e-12)
        i_na = s.g_na * m_na_inf ^ 3 * (1.0 - s.n) * (s.v - s.e_na)
        i_nap = s.g_nap * m_nap_inf * s.h_nap * (s.v - s.e_na)
        i_k = s.g_k * s.n ^ 4 * (s.v - s.e_k)
        i_l = s.g_l * (s.v - s.e_l)
        s.v += (-i_na - i_nap - i_k - i_l + I_ext) * s.dt
        s.v = Float64(clamp(s.v, -200, 100))
        s.n += (n_inf - s.n) / max(tau_n, 0.01) * s.dt
        s.n = Float64(clamp(s.n, 0, 1))
        s.h_nap += (h_nap_inf - s.h_nap) / max(tau_h, 0.1) * s.dt
        s.h_nap = Float64(clamp(s.h_nap, 0, 1))
        return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = ButeraRespiratoryNeuronState()
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

end # module ButeraRespiratoryAccel
