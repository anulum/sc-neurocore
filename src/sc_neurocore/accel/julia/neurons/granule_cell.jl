# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for granule_cell

module GranuleCellAccel

export step!, simulate, GranuleCellState

mutable struct GranuleCellState
    v::Float64
    m::Float64
    h::Float64
    n::Float64
    a::Float64
    b::Float64
    m_t::Float64
    s::Float64
    ca::Float64
    r::Float64
    c_m::Float64
    g_na::Float64
    g_kdr::Float64
    g_ka::Float64
    g_t::Float64
    g_kca::Float64
    g_h::Float64
    g_l::Float64
    g_tonic::Float64
    e_na::Float64
    e_k::Float64
    e_ca::Float64
    e_h::Float64
    e_l::Float64
    e_gaba::Float64
    tau_ca::Float64
    kd_kca::Float64
    dt::Float64
    sub_steps::Int64
    gain::Float64
end

function GranuleCellState()
    return GranuleCellState(
        -70.0, 0.02, 0.85, 0.05, 0.1, 0.8, 0.01, 0.95, 0.05, 0.1,
        1.0, 17.0, 9.0, 1.0, 0.5, 3.5, 0.03, 0.1, 0.2,
        87.4, -84.7, 129.3, -40.0, -58.0, -75.0, 10.0, 0.2, 0.5, 4, 1.0,
    )
end

function _boltz(v::Float64, vh::Float64, k::Float64)
    z = -(v - vh) / k
    if z > 60.0
        return 0.0
    elseif z < -60.0
        return 1.0
    end
    return 1.0 / (1.0 + exp(z))
end

_clamp01(x::Float64) = max(0.0, min(1.0, x))

function _validate(s::GranuleCellState)
    finite_values = (
        s.v, s.m, s.h, s.n, s.a, s.b, s.m_t, s.s, s.ca, s.r,
        s.c_m, s.g_na, s.g_kdr, s.g_ka, s.g_t, s.g_kca, s.g_h, s.g_l, s.g_tonic,
        s.e_na, s.e_k, s.e_ca, s.e_h, s.e_l, s.e_gaba, s.tau_ca, s.kd_kca, s.dt, s.gain,
    )
    all(isfinite, finite_values) || throw(ArgumentError("granule cell state and parameters must be finite"))
    all(x -> 0.0 <= x <= 1.0, (s.m, s.h, s.n, s.a, s.b, s.m_t, s.s, s.r)) ||
        throw(ArgumentError("granule cell gates must stay in [0, 1]"))
    s.ca >= 0.0 || throw(ArgumentError("granule cell calcium concentration must be non-negative"))
    all(x -> x >= 0.0, (s.g_na, s.g_kdr, s.g_ka, s.g_t, s.g_kca, s.g_h, s.g_l, s.g_tonic)) ||
        throw(ArgumentError("granule cell conductances must be non-negative"))
    (s.c_m > 0.0 && s.tau_ca > 0.0 && s.kd_kca > 0.0 && s.dt > 0.0) ||
        throw(ArgumentError("granule cell capacitance, calcium, and timestep parameters must be positive"))
    s.sub_steps > 0 || throw(ArgumentError("granule cell sub_steps must be positive"))
    s.gain >= 0.0 || throw(ArgumentError("granule cell gain must be non-negative"))
    return nothing
end

function step!(s::GranuleCellState, I_ext::Float64=0.0; dt::Float64=s.dt)
    _validate(s)
    isfinite(I_ext) || throw(ArgumentError("current must be finite"))
    (isfinite(dt) && dt > 0.0) || throw(ArgumentError("dt must be finite and positive"))

    inp = s.gain * I_ext
    dt_sub = dt / Float64(s.sub_steps)
    v_prev = s.v
    v = s.v
    m = s.m
    h = s.h
    n = s.n
    a = s.a
    b = s.b
    m_t = s.m_t
    gate_s = s.s
    ca = s.ca
    r = s.r

    for _ in 1:s.sub_steps
        m_inf = _boltz(v, -30.0, 7.0)
        tau_m = 0.1 + 0.3 / max(0.01, 1.0 + ((v + 30.0) / 10.0)^2)
        m = _clamp01(m + dt_sub * (m_inf - m) / tau_m)

        h_inf = _boltz(v, -52.0, -6.0)
        tau_h = 0.5 + 5.0 / max(0.01, 1.0 + ((v + 50.0) / 15.0)^2)
        h = _clamp01(h + dt_sub * (h_inf - h) / tau_h)

        n_inf = _boltz(v, -35.0, 8.0)
        tau_n = 1.0 + 5.0 / max(0.01, 1.0 + ((v + 35.0) / 15.0)^2)
        n = _clamp01(n + dt_sub * (n_inf - n) / tau_n)

        a_inf = _boltz(v, -50.0, 20.0)
        a = _clamp01(a + dt_sub * (a_inf - a) / 2.0)

        b_inf = _boltz(v, -70.0, -6.0)
        b = _clamp01(b + dt_sub * (b_inf - b) / 50.0)

        mt_inf = _boltz(v, -52.0, 5.0)
        m_t = _clamp01(m_t + dt_sub * (mt_inf - m_t))

        s_inf = _boltz(v, -60.0, -6.5)
        tau_s = 20.0 + 50.0 / max(0.01, 1.0 + ((v + 65.0) / 10.0)^2)
        gate_s = _clamp01(gate_s + dt_sub * (s_inf - gate_s) / tau_s)

        r_inf = _boltz(v, -80.0, -10.0)
        tau_r = 50.0 + 200.0 / max(0.01, 1.0 + ((v + 80.0) / 20.0)^2)
        r = _clamp01(r + dt_sub * (r_inf - r) / tau_r)

        i_ca_t = s.g_t * m_t^2 * gate_s * (v - s.e_ca)
        ca_entry = i_ca_t < 0.0 ? -i_ca_t * 0.001 : 0.0
        ca = max(0.0, ca + dt_sub * (-ca / s.tau_ca + ca_entry))

        kca_inf = ca^2 / (ca^2 + s.kd_kca^2)
        i_na = s.g_na * m^3 * h * (v - s.e_na)
        i_kdr = s.g_kdr * n^4 * (v - s.e_k)
        i_ka = s.g_ka * a^3 * b * (v - s.e_k)
        i_kca = s.g_kca * kca_inf * (v - s.e_k)
        i_h = s.g_h * r * (v - s.e_h)
        i_l = s.g_l * (v - s.e_l)
        i_gaba = s.g_tonic * (v - s.e_gaba)

        dv_val = (-(i_na + i_kdr + i_ka + i_ca_t + i_kca + i_h + i_l + i_gaba) + inp) / s.c_m
        v = max(-100.0, min(60.0, v + dt_sub * dv_val))

        all(isfinite, (v, m, h, n, a, b, m_t, gate_s, ca, r)) ||
            throw(ArgumentError("granule cell integration produced non-finite state"))
    end

    s.v = v
    s.m = m
    s.h = h
    s.n = n
    s.a = a
    s.b = b
    s.m_t = m_t
    s.s = gate_s
    s.ca = ca
    s.r = r
    return (s.v >= 0.0 && v_prev < 0.0) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.5)
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative"))
    s = GranuleCellState()
    s.dt = dt
    _validate(s)
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

end # module GranuleCellAccel
