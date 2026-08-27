# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Retained three-state project ion-mass recurrence

module SCDecoupledAdaptationIonMassAccel

export step!, reset!, SCDecoupledAdaptationIonMassNeuronState

mutable struct SCDecoupledAdaptationIonMassNeuronState
    v::Float64; w::Float64; z::Float64
    g_ca::Float64; g_na::Float64; g_k::Float64; g_l::Float64
    v_ca::Float64; v_na::Float64; v_k::Float64; v_l::Float64
    phi::Float64; tau_k::Float64; b::Float64; a_ee::Float64
    v0::Float64; i_ext::Float64; dt::Float64
end

SCDecoupledAdaptationIonMassNeuronState() = SCDecoupledAdaptationIonMassNeuronState(
    -0.5, 0.0, 0.0, 1.1, 6.7, 2.0, 0.5, 1.0, 0.53, -0.7, -0.5,
    0.7, 1.0, 0.1, 0.36, 0.0, 0.3, 0.01,
)

_gate(v, midpoint, width) = 0.5 * (1.0 + tanh((v - midpoint) / width))

function _derivatives(s, v, w, z, coupling)
    m_ca = _gate(v, -0.01, 0.15)
    m_na = _gate(v, 0.12, 0.15)
    m_k = _gate(v, s.v0, 0.3)
    dv = -s.g_ca*m_ca*(v-s.v_ca) - s.g_na*m_na*(v-s.v_na) -
        s.g_k*w*(v-s.v_k) - s.g_l*(v-s.v_l) + s.i_ext + coupling + s.a_ee*v
    (dv, s.phi*(m_k-w)/s.tau_k, s.b*(v+0.5-z))
end

function _valid(s)
    values = (s.v, s.w, s.z, s.g_ca, s.g_na, s.g_k, s.g_l, s.v_ca,
        s.v_na, s.v_k, s.v_l, s.phi, s.tau_k, s.b, s.a_ee, s.v0, s.i_ext, s.dt)
    all(isfinite, values) && 0.0 <= s.w <= 1.0 &&
        all(x -> x > 0.0, (s.g_ca, s.g_na, s.g_k, s.g_l, s.phi, s.tau_k, s.b, s.dt))
end

function step!(s::SCDecoupledAdaptationIonMassNeuronState, coupling::Float64=0.0)
    isfinite(coupling) && _valid(s) || throw(ArgumentError("invalid SC ion-mass input"))
    v, w, z, dt = s.v, s.w, s.z, s.dt
    k1 = _derivatives(s, v, w, z, coupling)
    k2 = _derivatives(s, v+0.5dt*k1[1], w+0.5dt*k1[2], z+0.5dt*k1[3], coupling)
    k3 = _derivatives(s, v+0.5dt*k2[1], w+0.5dt*k2[2], z+0.5dt*k2[3], coupling)
    k4 = _derivatives(s, v+dt*k3[1], w+dt*k3[2], z+dt*k3[3], coupling)
    candidate = (v+dt*(k1[1]+2k2[1]+2k3[1]+k4[1])/6,
        w+dt*(k1[2]+2k2[2]+2k3[2]+k4[2])/6,
        z+dt*(k1[3]+2k2[3]+2k3[3]+k4[3])/6)
    all(isfinite, candidate) && 0.0 <= candidate[2] <= 1.0 || throw(ArgumentError("invalid SC ion-mass candidate"))
    s.v, s.w, s.z = candidate
    s.v
end

function reset!(s::SCDecoupledAdaptationIonMassNeuronState)
    s.v, s.w, s.z = -0.5, 0.0, 0.0
    nothing
end

end
