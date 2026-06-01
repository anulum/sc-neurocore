# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia GLIF RK4 dynamics

module GlifAccel

export step!, simulate, GLIFNeuronState, valid_runtime

mutable struct GLIFNeuronState
    v::Float64
    theta::Float64
    theta_inf::Float64
    i_asc1::Float64
    i_asc2::Float64
    v_rest::Float64
    v_reset::Float64
    tau_m::Float64
    tau_theta::Float64
    tau_asc1::Float64
    tau_asc2::Float64
    a_theta::Float64
    delta_theta::Float64
    r_asc1::Float64
    r_asc2::Float64
    resistance::Float64
    dt::Float64
end

function GLIFNeuronState()
    GLIFNeuronState(-70.0, -50.0, -50.0, 0.0, 0.0, -70.0, -70.0, 10.0, 100.0, 10.0, 200.0, 0.01, 2.0, 1.0, 0.5, 1.0, 1.0)
end

finite_values(values...) = all(isfinite, values)

function valid_runtime(s::GLIFNeuronState)::Bool
    finite_values(
        s.v, s.theta, s.theta_inf, s.i_asc1, s.i_asc2, s.v_rest, s.v_reset,
        s.tau_m, s.tau_theta, s.tau_asc1, s.tau_asc2, s.a_theta, s.delta_theta,
        s.r_asc1, s.r_asc2, s.resistance, s.dt,
    ) && s.tau_m > 0.0 && s.tau_theta > 0.0 && s.tau_asc1 > 0.0 && s.tau_asc2 > 0.0 && s.dt > 0.0 && s.delta_theta >= 0.0 && s.resistance >= 0.0
end

function derivatives(s::GLIFNeuronState, v::Float64, theta::Float64, i_asc1::Float64, i_asc2::Float64, I_ext::Float64)
    (
        (-(v - s.v_rest) + s.resistance * I_ext + i_asc1 + i_asc2) / s.tau_m,
        (s.theta_inf - theta + s.a_theta * (v - s.v_rest)) / s.tau_theta,
        -i_asc1 / s.tau_asc1,
        -i_asc2 / s.tau_asc2,
    )
end

function add_scaled(state::NTuple{4,Float64}, slope::NTuple{4,Float64}, scale::Float64)
    ntuple(index -> state[index] + scale * slope[index], 4)
end

function rk4_candidate(s::GLIFNeuronState, I_ext::Float64)
    state = (s.v, s.theta, s.i_asc1, s.i_asc2)
    half_dt = 0.5 * s.dt
    k1 = derivatives(s, state..., I_ext)
    k2 = derivatives(s, add_scaled(state, k1, half_dt)..., I_ext)
    k3 = derivatives(s, add_scaled(state, k2, half_dt)..., I_ext)
    k4 = derivatives(s, add_scaled(state, k3, s.dt)..., I_ext)
    candidate = ntuple(index -> state[index] + s.dt * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]) / 6.0, 4)
    return candidate, finite_values(candidate...)
end

function step!(s::GLIFNeuronState, I_ext::Float64=0.0)
    if !isfinite(I_ext) || !valid_runtime(s)
        return 0
    end
    candidate, ok = rk4_candidate(s, I_ext)
    if !ok
        return 0
    end
    s.v, s.theta, s.i_asc1, s.i_asc2 = candidate
    if s.v >= s.theta
        s.v = s.v_reset
        s.theta += s.delta_theta
        s.i_asc1 += s.r_asc1
        s.i_asc2 += s.r_asc2
        return 1
    end
    return 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0)
    s = GLIFNeuronState()
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

end # module GlifAccel
