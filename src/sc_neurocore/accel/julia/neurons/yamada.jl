# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for yamada

module YamadaAccel

export step!, simulate, YamadaNeuronState, valid, reset!

mutable struct YamadaNeuronState
    v::Float64
    n::Float64
    q::Float64
    g_na::Float64
    g_k::Float64
    g_q::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_q::Float64
    e_l::Float64
    tau_q::Float64
    dt::Float64
    v_threshold::Float64
end

function YamadaNeuronState()
    YamadaNeuronState(-60.0, 0.1, 0.0, 20.0, 10.0, 5.0, 0.5, 60.0, -80.0, -80.0, -60.0, 300.0, 0.05, -20.0)
end

_sigmoid(x::Float64)::Float64 = x >= 0.0 ? 1.0 / (1.0 + exp(-x)) : (let z = exp(x); z / (1.0 + z) end)

function valid(s::YamadaNeuronState)::Bool
    return isfinite(s.v) &&
        isfinite(s.n) && 0.0 <= s.n <= 1.0 &&
        isfinite(s.q) && 0.0 <= s.q <= 1.0 &&
        isfinite(s.g_na) && s.g_na >= 0.0 &&
        isfinite(s.g_k) && s.g_k >= 0.0 &&
        isfinite(s.g_q) && s.g_q >= 0.0 &&
        isfinite(s.g_l) && s.g_l >= 0.0 &&
        isfinite(s.e_na) && isfinite(s.e_k) && isfinite(s.e_q) && isfinite(s.e_l) &&
        isfinite(s.tau_q) && s.tau_q > 0.0 &&
        isfinite(s.dt) && s.dt > 0.0 &&
        isfinite(s.v_threshold)
end

function step!(s::YamadaNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)::Int
    if !isfinite(I_ext) || !isfinite(dt) || dt <= 0.0 || !valid(s)
        return 0
    end

    v_prev = s.v
    m_inf = _sigmoid((s.v + 30.0) / 9.5)
    n_inf = _sigmoid((s.v + 30.0) / 10.0)
    q_inf = _sigmoid((s.v + 50.0) / 10.0)
    tau_n = 1.0 + 7.5 / (1.0 + exp((s.v + 40.0) / 12.0))
    i_na = s.g_na * m_inf ^ 3 * (1.0 - s.n) * (s.v - s.e_na)
    i_k = s.g_k * s.n ^ 4 * (s.v - s.e_k)
    i_q = s.g_q * s.q * (s.v - s.e_q)
    i_l = s.g_l * (s.v - s.e_l)
    dv = (-i_na - i_k - i_q - i_l + I_ext) * dt
    dn = (n_inf - s.n) / tau_n * dt
    dq = (q_inf - s.q) / s.tau_q * dt
    next_v = s.v + dv
    next_n = s.n + dn
    next_q = s.q + dq
    if any(!isfinite, (m_inf, n_inf, q_inf, tau_n, i_na, i_k, i_q, i_l, dv, dn, dq, next_v, next_n, next_q)) || !(0.0 <= next_n <= 1.0) || !(0.0 <= next_q <= 1.0)
        return 0
    end

    s.dt = dt
    s.v = next_v
    s.n = next_n
    s.q = next_q
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function reset!(s::YamadaNeuronState)::Nothing
    s.v = -60.0
    s.n = 0.1
    s.q = 0.0
    return nothing
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.05)
    s = YamadaNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module YamadaAccel
