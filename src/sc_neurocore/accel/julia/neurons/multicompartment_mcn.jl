# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia candidate-first RK4 for multicompartment_mcn

module MulticompartmentMcnAccel

export step!, step_compartments!, simulate, MulticompartmentMCNNeuronState

mutable struct MulticompartmentMCNNeuronState
    tau::Float64
    tau_b::Float64
    tau_a::Float64
    g_ratio::Float64
    beta::Float64
    v_th::Float64
    dt::Float64
    u::Float64
    v_basal::Float64
    v_apical::Float64
end

function MulticompartmentMCNNeuronState()
    MulticompartmentMCNNeuronState(2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0)
end

@inline function _finite(values...)
    return all(isfinite, values)
end

@inline function _valid(s::MulticompartmentMCNNeuronState)
    return _finite(s.tau, s.tau_b, s.tau_a, s.g_ratio, s.beta, s.v_th, s.dt, s.u, s.v_basal, s.v_apical) &&
        s.tau > 0.0 && s.tau_b > 0.0 && s.tau_a > 0.0 && s.g_ratio >= 0.0 &&
        s.beta > 0.0 && s.v_th > 0.0 && s.dt > 0.0
end

@inline function _sigma(s::MulticompartmentMCNNeuronState, x::Float64)
    return 1.0 / (1.0 + exp(-s.beta * x))
end

@inline function _derivatives(
    s::MulticompartmentMCNNeuronState,
    u::Float64,
    v_basal::Float64,
    v_apical::Float64,
    x_basal::Float64,
    x_apical::Float64,
    i_soma::Float64,
)
    gate = _sigma(s, v_apical)
    du = (-u + gate * (s.g_ratio * (v_basal - u) + i_soma)) / s.tau
    dv_basal = (-v_basal + x_basal) / s.tau_b
    dv_apical = (-v_apical + x_apical) / s.tau_a
    return du, dv_basal, dv_apical
end

@inline function _rk4_substep(
    s::MulticompartmentMCNNeuronState,
    u::Float64,
    v_basal::Float64,
    v_apical::Float64,
    x_basal::Float64,
    x_apical::Float64,
    i_soma::Float64,
)
    dt = s.dt
    k1u, k1vb, k1va = _derivatives(s, u, v_basal, v_apical, x_basal, x_apical, i_soma)
    k2u, k2vb, k2va = _derivatives(
        s,
        u + 0.5 * dt * k1u,
        v_basal + 0.5 * dt * k1vb,
        v_apical + 0.5 * dt * k1va,
        x_basal,
        x_apical,
        i_soma,
    )
    k3u, k3vb, k3va = _derivatives(
        s,
        u + 0.5 * dt * k2u,
        v_basal + 0.5 * dt * k2vb,
        v_apical + 0.5 * dt * k2va,
        x_basal,
        x_apical,
        i_soma,
    )
    k4u, k4vb, k4va = _derivatives(
        s,
        u + dt * k3u,
        v_basal + dt * k3vb,
        v_apical + dt * k3va,
        x_basal,
        x_apical,
        i_soma,
    )
    next_u = u + dt * (k1u + 2.0 * k2u + 2.0 * k3u + k4u) / 6.0
    next_v_basal = v_basal + dt * (k1vb + 2.0 * k2vb + 2.0 * k3vb + k4vb) / 6.0
    next_v_apical = v_apical + dt * (k1va + 2.0 * k2va + 2.0 * k3va + k4va) / 6.0
    return next_u, next_v_basal, next_v_apical
end

function step_compartments!(
    s::MulticompartmentMCNNeuronState,
    x_basal::Float64,
    x_apical::Float64,
    i_soma::Float64,
)
    if !_finite(x_basal, x_apical, i_soma) || !_valid(s)
        return 0
    end
    next_u, next_v_basal, next_v_apical = _rk4_substep(
        s,
        s.u,
        s.v_basal,
        s.v_apical,
        x_basal,
        x_apical,
        i_soma,
    )
    if !_finite(next_u, next_v_basal, next_v_apical)
        return 0
    end
    spike = next_u >= s.v_th ? 1 : 0
    s.u = spike == 1 ? 0.0 : next_u
    s.v_basal = next_v_basal
    s.v_apical = next_v_apical
    if spike == 1
        s.u = 0.0
        return 1
    end
    return 0
end

function step!(s::MulticompartmentMCNNeuronState, I_ext::Float64=0.0; dt::Float64=1.0)
    return step_compartments!(s, I_ext, 0.0, 0.0)
end

function simulate(n_steps::Int=1000; I_ext::Float64=3.2, dt::Float64=1.0)
    s = MulticompartmentMCNNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.u
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module MulticompartmentMcnAccel
