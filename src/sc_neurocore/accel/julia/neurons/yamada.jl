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

function _tau_n(v::Float64)::Float64
    x = (v + 40.0) / 12.0
    if !isfinite(x)
        return NaN
    elseif x > 709.0
        return 1.0
    end
    return 1.0 + 7.5 / (1.0 + exp(x))
end

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

function _derivatives(s::YamadaNeuronState, v::Float64, n::Float64, q::Float64, I_ext::Float64)
    if any(!isfinite, (v, n, q, I_ext)) || !(0.0 <= n <= 1.0) || !(0.0 <= q <= 1.0)
        return 0.0, 0.0, 0.0, false
    end
    m_inf = _sigmoid((v + 30.0) / 9.5)
    n_inf = _sigmoid((v + 30.0) / 10.0)
    q_inf = _sigmoid((v + 50.0) / 10.0)
    tau_n = _tau_n(v)
    i_na = s.g_na * m_inf ^ 3 * (1.0 - n) * (v - s.e_na)
    i_k = s.g_k * n ^ 4 * (v - s.e_k)
    i_q = s.g_q * q * (v - s.e_q)
    i_l = s.g_l * (v - s.e_l)
    dv = -i_na - i_k - i_q - i_l + I_ext
    dn = (n_inf - n) / tau_n
    dq = (q_inf - q) / s.tau_q
    if any(!isfinite, (m_inf, n_inf, q_inf, tau_n, i_na, i_k, i_q, i_l, dv, dn, dq))
        return 0.0, 0.0, 0.0, false
    end
    return dv, dn, dq, true
end

function _rk4_candidate(s::YamadaNeuronState, I_ext::Float64, dt::Float64)
    if !isfinite(I_ext) || !isfinite(dt) || dt <= 0.0 || !valid(s)
        return 0.0, 0.0, 0.0, false
    end
    v0, n0, q0 = s.v, s.n, s.q
    k1v, k1n, k1q, ok = _derivatives(s, v0, n0, q0, I_ext)
    ok || return 0.0, 0.0, 0.0, false
    k2v, k2n, k2q, ok = _derivatives(s, v0 + 0.5 * dt * k1v, n0 + 0.5 * dt * k1n, q0 + 0.5 * dt * k1q, I_ext)
    ok || return 0.0, 0.0, 0.0, false
    k3v, k3n, k3q, ok = _derivatives(s, v0 + 0.5 * dt * k2v, n0 + 0.5 * dt * k2n, q0 + 0.5 * dt * k2q, I_ext)
    ok || return 0.0, 0.0, 0.0, false
    k4v, k4n, k4q, ok = _derivatives(s, v0 + dt * k3v, n0 + dt * k3n, q0 + dt * k3q, I_ext)
    ok || return 0.0, 0.0, 0.0, false

    next_v = v0 + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
    next_n = n0 + dt * (k1n + 2.0 * k2n + 2.0 * k3n + k4n) / 6.0
    next_q = q0 + dt * (k1q + 2.0 * k2q + 2.0 * k3q + k4q) / 6.0
    if any(!isfinite, (next_v, next_n, next_q)) || !(0.0 <= next_n <= 1.0) || !(0.0 <= next_q <= 1.0)
        return 0.0, 0.0, 0.0, false
    end
    return next_v, next_n, next_q, true
end

function step!(s::YamadaNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)::Int
    if !isfinite(I_ext) || !isfinite(dt) || dt <= 0.0 || !valid(s)
        return 0
    end

    v_prev = s.v
    next_v, next_n, next_q, ok = _rk4_candidate(s, I_ext, dt)
    if !ok
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
