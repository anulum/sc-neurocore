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

function step!(s::LarterBreakspearNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        i_ca = s.g_ca * s._m_ca(s.v) * (s.v - s.v_ca)
        i_na = s.g_na * s._m_na(s.v) * (s.v - s.v_na)
        i_k = s.g_k * s.w * (s.v - s.v_k)
        i_l = s.g_l * (s.v - s.v_l)
        dv = -i_ca - i_na - i_k - i_l + s.i_ext + coupling + s.a_ee * s.v
        dw = s.phi * (s._m_k(s.v) - s.w) / s.tau_k
        dz = s.b * (s.v + 0.5 - s.z)
        s.v += dv * s.dt
        s.w += dw * s.dt
        s.z += dz * s.dt
        return s.v
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = LarterBreakspearNeuronState()
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

end # module LarterBreakspearAccel
