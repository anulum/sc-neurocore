# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia candidate-first RK4 for dendritic NMDA

module DendriticNmdaAccel

export step!, simulate, DendriticNMDANeuronState

mutable struct DendriticNMDANeuronState
    g_nmda::Float64
    e_nmda::Float64
    mg_conc::Float64
    g_coupling::Float64
    tau_soma::Float64
    tau_dend::Float64
    theta::Float64
    dt::Float64
    v_soma::Float64
    v_dend::Float64
end

function DendriticNMDANeuronState()
    DendriticNMDANeuronState(1.5, 0.0, 1.0, 0.5, 20.0, 50.0, -50.0, 0.1, -65.0, -65.0)
end

@inline function _finite(values...)
    return all(isfinite, values)
end

@inline function _valid(s::DendriticNMDANeuronState)
    return _finite(s.g_nmda, s.e_nmda, s.mg_conc, s.g_coupling, s.tau_soma, s.tau_dend, s.theta, s.dt, s.v_soma, s.v_dend) &&
        s.g_nmda >= 0.0 && s.mg_conc >= 0.0 && s.g_coupling >= 0.0 &&
        s.tau_soma > 0.0 && s.tau_dend > 0.0 && s.dt > 0.0
end

@inline function mg_block(s::DendriticNMDANeuronState, v::Float64)
    return 1.0 / (1.0 + (s.mg_conc / 3.57) * exp(-0.062 * v))
end

@inline function _derivatives(s::DendriticNMDANeuronState, v_soma::Float64, v_dend::Float64, i_soma::Float64, glutamate::Float64)
    block = mg_block(s, v_dend)
    i_nmda = s.g_nmda * glutamate * block * (v_dend - s.e_nmda)
    dv_soma = (-v_soma - 65.0 + i_soma + s.g_coupling * (v_dend - v_soma)) / s.tau_soma
    dv_dend = (-v_dend - 65.0 + i_nmda + s.g_coupling * (v_soma - v_dend)) / s.tau_dend
    return dv_soma, dv_dend
end

@inline function _rk4_substep(s::DendriticNMDANeuronState, v_soma::Float64, v_dend::Float64, i_soma::Float64, glutamate::Float64)
    dt = s.dt
    k1s, k1d = _derivatives(s, v_soma, v_dend, i_soma, glutamate)
    k2s, k2d = _derivatives(s, v_soma + 0.5 * dt * k1s, v_dend + 0.5 * dt * k1d, i_soma, glutamate)
    k3s, k3d = _derivatives(s, v_soma + 0.5 * dt * k2s, v_dend + 0.5 * dt * k2d, i_soma, glutamate)
    k4s, k4d = _derivatives(s, v_soma + dt * k3s, v_dend + dt * k3d, i_soma, glutamate)
    return (
        v_soma + dt * (k1s + 2.0 * k2s + 2.0 * k3s + k4s) / 6.0,
        v_dend + dt * (k1d + 2.0 * k2d + 2.0 * k3d + k4d) / 6.0,
    )
end

function step!(s::DendriticNMDANeuronState, i_soma::Float64=0.0, glutamate::Float64=0.0)
    if !_finite(i_soma, glutamate) || glutamate < 0.0 || !_valid(s)
        return 0
    end
    next_v_soma, next_v_dend = _rk4_substep(s, s.v_soma, s.v_dend, i_soma, glutamate)
    if !_finite(next_v_soma, next_v_dend)
        return 0
    end
    s.v_dend = next_v_dend
    if next_v_soma >= s.theta
        s.v_soma = -65.0
        return 1
    end
    s.v_soma = next_v_soma
    return 0
end

function simulate(n_steps::Int=1000; i_soma::Float64=50.0, glutamate::Float64=0.5)
    s = DendriticNMDANeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, i_soma, glutamate)
        trace[t] = s.v_soma
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module DendriticNmdaAccel
