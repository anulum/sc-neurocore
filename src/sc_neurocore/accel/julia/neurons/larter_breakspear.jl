# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Larter-Breakspear source dynamics

module LarterBreakspearAccel

export step!, simulate, reset!, valid, LarterBreakspearNeuronState

"""Complete state and source-profile configuration for the cortical neural mass."""
mutable struct LarterBreakspearNeuronState
    v::Float64; w::Float64; z::Float64
    g_ca::Float64; g_na::Float64; g_k::Float64; g_l::Float64
    v_ca::Float64; v_na::Float64; v_k::Float64; v_l::Float64
    t_ca::Float64; t_na::Float64; t_k::Float64
    delta_ca::Float64; delta_na::Float64; delta_k::Float64
    phi::Float64; tau_k::Float64; b::Float64
    a_ee::Float64; a_ei::Float64; a_ie::Float64; a_ne::Float64; a_ni::Float64
    r_nmda::Float64; coupling_balance::Float64
    v_t::Float64; z_t::Float64; delta_v::Float64; delta_z::Float64
    q_v_max::Float64; q_z_max::Float64; i_ext::Float64; t_scale::Float64; dt::Float64
end

function LarterBreakspearNeuronState()
    LarterBreakspearNeuronState(
        0.1, 0.1, 0.1, 1.1, 6.7, 2.0, 0.5, 1.0, 0.53, -0.7, -0.5,
        -0.01, 0.3, 0.0, 0.15, 0.15, 0.3, 0.7, 1.0, 0.1,
        0.4, 2.0, 2.0, 1.0, 0.4, 0.25, 0.1, 0.0, 0.0, 0.65, 0.7,
        1.0, 1.0, 0.3, 1.0, 0.01,
    )
end

_sigmoid(value, threshold, width) = 0.5 * (1.0 + tanh((value - threshold) / width))

"""Return true only for finite state and a physically admissible configuration."""
function valid(s::LarterBreakspearNeuronState)
    values = (s.v, s.w, s.z, s.g_ca, s.g_na, s.g_k, s.g_l, s.v_ca, s.v_na,
        s.v_k, s.v_l, s.t_ca, s.t_na, s.t_k, s.delta_ca, s.delta_na, s.delta_k,
        s.phi, s.tau_k, s.b, s.a_ee, s.a_ei, s.a_ie, s.a_ne, s.a_ni, s.r_nmda,
        s.coupling_balance, s.v_t, s.z_t, s.delta_v, s.delta_z, s.q_v_max,
        s.q_z_max, s.i_ext, s.t_scale, s.dt)
    all(isfinite, values) && 0.0 <= s.w <= 1.0 &&
        all(value -> value >= 0.0, (s.g_ca, s.g_na, s.g_k, s.g_l, s.r_nmda, s.q_v_max, s.q_z_max)) &&
        all(value -> value > 0.0, (s.delta_ca, s.delta_na, s.delta_k, s.delta_v, s.delta_z, s.phi, s.tau_k, s.b, s.t_scale, s.dt)) &&
        0.0 <= s.coupling_balance <= 1.0
end

function _derivatives(s::LarterBreakspearNeuronState, v, w, z, coupling)
    m_ca = _sigmoid(v, s.t_ca, s.delta_ca)
    m_na = _sigmoid(v, s.t_na, s.delta_na)
    m_k = _sigmoid(v, s.t_k, s.delta_k)
    q_v = s.q_v_max * _sigmoid(v, s.v_t, s.delta_v)
    q_z = s.q_z_max * _sigmoid(z, s.z_t, s.delta_z)
    excitation = s.a_ee * ((1.0 - s.coupling_balance) * q_v + s.coupling_balance * coupling)
    dv = -(s.g_ca + s.r_nmda * excitation) * m_ca * (v - s.v_ca) -
        s.g_k * w * (v - s.v_k) - s.g_l * (v - s.v_l) -
        (s.g_na * m_na + excitation) * (v - s.v_na) - s.a_ie * z * q_z + s.a_ne * s.i_ext
    dw = s.phi * (m_k - w) / s.tau_k
    dz = s.b * (s.a_ni * s.i_ext + s.a_ei * v * q_v)
    s.t_scale .* (dv, dw, dz)
end

"""Advance one fixed-step classical-RK4 transition atomically."""
function step!(s::LarterBreakspearNeuronState, coupling::Float64=0.0; dt::Float64=s.dt)
    isfinite(coupling) || throw(ArgumentError("coupling must be finite"))
    valid(s) || throw(ArgumentError("invalid Larter-Breakspear state or configuration"))
    dt == s.dt || throw(ArgumentError("dt must match the configured step"))
    v0, w0, z0 = s.v, s.w, s.z
    k1 = _derivatives(s, v0, w0, z0, coupling)
    k2 = _derivatives(s, v0 + 0.5dt*k1[1], w0 + 0.5dt*k1[2], z0 + 0.5dt*k1[3], coupling)
    k3 = _derivatives(s, v0 + 0.5dt*k2[1], w0 + 0.5dt*k2[2], z0 + 0.5dt*k2[3], coupling)
    k4 = _derivatives(s, v0 + dt*k3[1], w0 + dt*k3[2], z0 + dt*k3[3], coupling)
    candidate = (
        v0 + dt*(k1[1] + 2k2[1] + 2k3[1] + k4[1])/6,
        w0 + dt*(k1[2] + 2k2[2] + 2k3[2] + k4[2])/6,
        z0 + dt*(k1[3] + 2k2[3] + 2k3[3] + k4[3])/6,
    )
    all(isfinite, candidate) || throw(ArgumentError("candidate became non-finite"))
    0.0 <= candidate[2] <= 1.0 || throw(ArgumentError("candidate potassium gate left [0, 1]"))
    s.v, s.w, s.z = candidate
    s.v
end

"""Restore source-profile dynamic state while retaining configuration."""
function reset!(s::LarterBreakspearNeuronState)
    s.v, s.w, s.z = 0.1, 0.1, 0.1
    nothing
end

"""Return the complete continuous state trace for a fresh source-profile mass."""
function simulate(n_steps::Int=1000; coupling::Float64=0.0, dt::Float64=0.01)
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative"))
    s = LarterBreakspearNeuronState()
    dt == s.dt || throw(ArgumentError("dt must match the configured step"))
    trace = Matrix{Float64}(undef, n_steps, 3)
    for index in 1:n_steps
        step!(s, coupling; dt=dt)
        trace[index, :] .= (s.v, s.w, s.z)
    end
    trace
end

end # module LarterBreakspearAccel
