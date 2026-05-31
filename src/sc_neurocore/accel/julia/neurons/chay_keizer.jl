# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for chay_keizer

module ChayKeizerAccel

export step!, simulate, ChayKeizerNeuronState, validate

const MAX_SUBSTEP = 0.001
const V_MIN = -200.0
const V_MAX = 200.0
const CA_MAX = 100.0

mutable struct ChayKeizerNeuronState
    v::Float64
    n::Float64
    ca::Float64
    g_ca::Float64
    g_k::Float64
    g_kca::Float64
    g_l::Float64
    e_ca::Float64
    e_k::Float64
    e_l::Float64
    k_d::Float64
    f_ca::Float64
    k_ca::Float64
    dt::Float64
    v_threshold::Float64
end

function ChayKeizerNeuronState()
    ChayKeizerNeuronState(-50.0, 0.01, 0.1, 20.0, 25.0, 12.0, 0.1, 100.0, -75.0, -40.0, 1.0, 0.004, 0.03, 0.02, -20.0)
end

_finite(value::Float64) = isfinite(value)
_probability(value::Float64) = isfinite(value) && 0.0 <= value <= 1.0
_nonnegative(value::Float64) = isfinite(value) && value >= 0.0

function _checked_exp(exponent::Float64)
    if !isfinite(exponent)
        error("exponent must be finite")
    elseif exponent < -700.0
        return 0.0
    elseif exponent > 700.0
        return exp(700.0)
    end
    return exp(exponent)
end

_gate_inf(exponent::Float64) = 1.0 / (1.0 + _checked_exp(exponent))

function validate(s::ChayKeizerNeuronState)
    if !_finite(s.v) || !(V_MIN <= s.v <= V_MAX)
        return false
    end
    if !_probability(s.n) || !_nonnegative(s.ca) || s.ca > CA_MAX
        return false
    end
    if any(x -> !_nonnegative(x), (s.g_ca, s.g_k, s.g_kca, s.g_l, s.f_ca, s.k_ca))
        return false
    end
    if !_finite(s.k_d) || s.k_d <= 0.0
        return false
    end
    if any(x -> !_finite(x), (s.e_ca, s.e_k, s.e_l, s.v_threshold))
        return false
    end
    return isfinite(s.dt) && s.dt > 0.0 && ceil(Int, s.dt / MAX_SUBSTEP) <= 10000
end

function _candidate(s::ChayKeizerNeuronState, v::Float64, n::Float64, ca::Float64, h::Float64, I_ext::Float64)
    m_inf = _gate_inf(-(v + 25.0) / 8.0)
    n_inf = _gate_inf(-(v + 18.0) / 14.0)
    tau_n = 20.0 / (1.0 + _checked_exp((v + 18.0) / 14.0))
    ca_denominator = ca + s.k_d
    if ca_denominator <= 0.0
        error("calcium activation denominator must be positive")
    end
    q_kca = ca / ca_denominator
    i_ca = s.g_ca * m_inf * (v - s.e_ca)
    i_k = s.g_k * n * (v - s.e_k)
    i_kca = s.g_kca * q_kca * (v - s.e_k)
    i_l = s.g_l * (v - s.e_l)

    v_next = v + (-i_ca - i_k - i_kca - i_l + I_ext) * h
    n_next = n + (n_inf - n) / max(tau_n, 0.1) * h
    ca_next = ca + (-s.f_ca * i_ca - s.k_ca * ca) * h

    if !isfinite(v_next) || !(V_MIN <= v_next <= V_MAX)
        error("Chay-Keizer voltage candidate outside safety envelope")
    end
    if !_probability(n_next)
        error("Chay-Keizer n-gate candidate outside [0, 1]")
    end
    if !_nonnegative(ca_next) || ca_next > CA_MAX
        error("Chay-Keizer calcium candidate outside safety envelope")
    end
    return v_next, n_next, ca_next
end

function step!(s::ChayKeizerNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)
    if !isfinite(I_ext) || !validate(s)
        return -1
    end
    substeps = max(1, ceil(Int, s.dt / MAX_SUBSTEP))
    h = s.dt / substeps
    v_initial = s.v
    v = s.v
    n = s.n
    ca = s.ca
    crossed = false
    try
        for _ in 1:substeps
            v_next, n_next, ca_next = _candidate(s, v, n, ca, h, I_ext)
            crossed = crossed || (v_next >= s.v_threshold && v < s.v_threshold)
            v, n, ca = v_next, n_next, ca_next
        end
    catch _e
        return -1
    end
    s.v = v
    s.n = n
    s.ca = ca
    return (crossed && v_initial < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = ChayKeizerNeuronState()
    s.dt = dt
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext)
        trace[t] = result < 0 ? NaN : s.v
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module ChayKeizerAccel
