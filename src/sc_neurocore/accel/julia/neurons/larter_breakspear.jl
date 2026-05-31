# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for larter_breakspear

module LarterBreakspearAccel

export step!, simulate, LarterBreakspearNeuronState

mutable struct LarterBreakspearNeuronState
    v::Float64
    w::Float64
    z::Float64
    g_ca::Float64
    g_na::Float64
    g_k::Float64
    v_ca::Float64
    v_na::Float64
    v_k::Float64
    v_l::Float64
    g_l::Float64
    phi::Float64
    tau_k::Float64
    b::Float64
    a_ee::Float64
    v0::Float64
    i_ext::Float64
    dt::Float64
end

function LarterBreakspearNeuronState()
    LarterBreakspearNeuronState(-0.5, 0.0, 0.0, 1.1, 6.7, 2.0, 1.0, 0.53, -0.7, -0.5, 0.5, 0.7, 1.0, 0.1, 0.36, 0.0, 0.3, 0.01)
end

function _m_ca(s::LarterBreakspearNeuronState, v)
    return 0.5 * (1.0 + tanh((v - -0.01) / 0.15))
end

function _m_na(s::LarterBreakspearNeuronState, v)
    return 0.5 * (1.0 + tanh((v - 0.12) / 0.15))
end

function _m_k(s::LarterBreakspearNeuronState, v)
    return 0.5 * (1.0 + tanh((v - s.v0) / 0.3))
end

function _valid(s::LarterBreakspearNeuronState)
    values = (
        s.v, s.w, s.z, s.g_ca, s.g_na, s.g_k, s.v_ca, s.v_na, s.v_k,
        s.v_l, s.g_l, s.phi, s.tau_k, s.b, s.a_ee, s.v0, s.i_ext, s.dt
    )
    return all(isfinite, values) &&
        s.dt > 0.0 &&
        s.tau_k > 0.0 &&
        s.phi > 0.0 &&
        s.b > 0.0 &&
        s.g_ca > 0.0 &&
        s.g_na > 0.0 &&
        s.g_k > 0.0 &&
        s.g_l > 0.0 &&
        0.0 <= s.w <= 1.0
end

function _derivatives(s::LarterBreakspearNeuronState, v::Float64, w::Float64, z::Float64, coupling::Float64)
    i_ca = s.g_ca * _m_ca(s, v) * (v - s.v_ca)
    i_na = s.g_na * _m_na(s, v) * (v - s.v_na)
    i_k = s.g_k * w * (v - s.v_k)
    i_l = s.g_l * (v - s.v_l)
    dv = -i_ca - i_na - i_k - i_l + s.i_ext + coupling + s.a_ee * v
    dw = s.phi * (_m_k(s, v) - w) / s.tau_k
    dz = s.b * (v + 0.5 - z)
    return dv, dw, dz
end

function step!(s::LarterBreakspearNeuronState, coupling::Float64=0.0; dt::Float64=s.dt)
    if !_valid(s) || !isfinite(coupling) || !isfinite(dt) || dt <= 0.0
        return NaN
    end

    v0, w0, z0 = s.v, s.w, s.z
    k1 = _derivatives(s, v0, w0, z0, coupling)
    k2 = _derivatives(s, v0 + 0.5 * dt * k1[1], w0 + 0.5 * dt * k1[2], z0 + 0.5 * dt * k1[3], coupling)
    k3 = _derivatives(s, v0 + 0.5 * dt * k2[1], w0 + 0.5 * dt * k2[2], z0 + 0.5 * dt * k2[3], coupling)
    k4 = _derivatives(s, v0 + dt * k3[1], w0 + dt * k3[2], z0 + dt * k3[3], coupling)

    candidate = LarterBreakspearNeuronState(
        v0 + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
        w0 + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
        z0 + (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]),
        s.g_ca, s.g_na, s.g_k, s.v_ca, s.v_na, s.v_k, s.v_l, s.g_l,
        s.phi, s.tau_k, s.b, s.a_ee, s.v0, s.i_ext, dt,
    )
    if !_valid(candidate)
        return NaN
    end
    s.v = candidate.v
    s.w = candidate.w
    s.z = candidate.z
    s.dt = candidate.dt
    return s.v
end

function simulate(n_steps::Int=1000; coupling::Float64=0.0, dt::Float64=0.01)
    s = LarterBreakspearNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, coupling; dt=dt)
        trace[t] = s.v
        if isnan(result)
            break
        end
    end
    return trace, spikes
end

end # module LarterBreakspearAccel
