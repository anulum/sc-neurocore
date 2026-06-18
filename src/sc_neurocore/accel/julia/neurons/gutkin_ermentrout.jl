# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for gutkin_ermentrout

module GutkinErmentroutAccel

export step!, simulate, GutkinErmentroutNeuronState

mutable struct GutkinErmentroutNeuronState
    v::Float64
    n::Float64
    g_na::Float64
    g_k::Float64
    g_l::Float64
    e_na::Float64
    e_k::Float64
    e_l::Float64
    dt::Float64
    v_threshold::Float64
end

function GutkinErmentroutNeuronState()
    GutkinErmentroutNeuronState(-65.0, 0.1, 20.0, 10.0, 8.0, 60.0, -90.0, -80.0, 0.05, -20.0)
end

finite_gutkin(x::Float64) = isfinite(x)

function validate(s::GutkinErmentroutNeuronState)
    return finite_gutkin(s.v) &&
        finite_gutkin(s.n) &&
        0.0 <= s.n <= 1.0 &&
        finite_gutkin(s.g_na) &&
        s.g_na >= 0.0 &&
        finite_gutkin(s.g_k) &&
        s.g_k >= 0.0 &&
        finite_gutkin(s.g_l) &&
        s.g_l >= 0.0 &&
        finite_gutkin(s.e_na) &&
        finite_gutkin(s.e_k) &&
        finite_gutkin(s.e_l) &&
        finite_gutkin(s.dt) &&
        s.dt > 0.0 &&
        finite_gutkin(s.v_threshold)
end

m_inf(v::Float64) = 1.0 / (1.0 + exp(-(v + 20.0) / 15.0))
n_inf(v::Float64) = 1.0 / (1.0 + exp(-(v + 25.0) / 5.0))

function rhs(s::GutkinErmentroutNeuronState, v::Float64, n_gate::Float64, I_ext::Float64)
    if !(finite_gutkin(v) && finite_gutkin(n_gate) && finite_gutkin(I_ext) && 0.0 <= n_gate <= 1.0)
        return nothing
    end
    m = m_inf(v)
    n_target = n_inf(v)
    if !(finite_gutkin(m) && finite_gutkin(n_target))
        return nothing
    end
    i_na = s.g_na * m * (v - s.e_na)
    i_k = s.g_k * n_gate * (v - s.e_k)
    i_l = s.g_l * (v - s.e_l)
    dv = -i_na - i_k - i_l + I_ext
    dn = n_target - n_gate
    return finite_gutkin(dv) && finite_gutkin(dn) ? (dv, dn) : nothing
end

function step!(s::GutkinErmentroutNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)
    if !(validate(s) && finite_gutkin(I_ext))
        return -1
    end
    if dt != s.dt
        s.dt = dt
        if !validate(s)
            return -1
        end
    end
    v_prev = s.v
    k1 = rhs(s, s.v, s.n, I_ext)
    k1 === nothing && return -1
    k2 = rhs(s, s.v + 0.5 * s.dt * k1[1], s.n + 0.5 * s.dt * k1[2], I_ext)
    k2 === nothing && return -1
    k3 = rhs(s, s.v + 0.5 * s.dt * k2[1], s.n + 0.5 * s.dt * k2[2], I_ext)
    k3 === nothing && return -1
    k4 = rhs(s, s.v + s.dt * k3[1], s.n + s.dt * k3[2], I_ext)
    k4 === nothing && return -1

    next_v = s.v + s.dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
    next_n = s.n + s.dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
    if !(finite_gutkin(next_v) && finite_gutkin(next_n) && 0.0 <= next_n <= 1.0)
        return -1
    end
    s.v = next_v
    s.n = next_n
    return (s.v >= s.v_threshold && v_prev < s.v_threshold) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = GutkinErmentroutNeuronState()
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

end # module GutkinErmentroutAccel
