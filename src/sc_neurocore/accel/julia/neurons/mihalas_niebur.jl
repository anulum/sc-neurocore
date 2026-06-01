# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Mihalas-Niebur RK4 dynamics

module MihalasNieburAccel

export step!, simulate, MihalasNieburNeuronState, valid_runtime

mutable struct MihalasNieburNeuronState
    v::Float64
    theta::Float64
    i1::Float64
    i2::Float64
    v_rest::Float64
    v_reset::Float64
    theta_reset::Float64
    theta_inf::Float64
    tau_v::Float64
    tau_theta::Float64
    tau_1::Float64
    tau_2::Float64
    a::Float64
    b::Float64
    r1::Float64
    r2::Float64
    dt::Float64
end

function MihalasNieburNeuronState()
    MihalasNieburNeuronState(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 10.0, 100.0, 10.0, 200.0, 0.0, 0.0, 0.0, 0.0, 1.0)
end

finite_values(values...) = all(isfinite, values)

function valid_runtime(s::MihalasNieburNeuronState)::Bool
    finite_values(
        s.v, s.theta, s.i1, s.i2, s.v_rest, s.v_reset, s.theta_reset, s.theta_inf,
        s.tau_v, s.tau_theta, s.tau_1, s.tau_2, s.a, s.b, s.r1, s.r2, s.dt,
    ) && s.tau_v > 0.0 && s.tau_theta > 0.0 && s.tau_1 > 0.0 && s.tau_2 > 0.0 && s.dt > 0.0
end

function derivatives(s::MihalasNieburNeuronState, v::Float64, theta::Float64, i1::Float64, i2::Float64, I_ext::Float64)
    (
        (-(v - s.v_rest) + i1 + i2 + I_ext) / s.tau_v,
        (s.theta_inf - theta + s.a * (v - s.v_rest)) / s.tau_theta,
        -i1 / s.tau_1,
        -i2 / s.tau_2,
    )
end

function add_scaled(state::NTuple{4,Float64}, slope::NTuple{4,Float64}, scale::Float64)
    ntuple(index -> state[index] + scale * slope[index], 4)
end

function rk4_candidate(s::MihalasNieburNeuronState, I_ext::Float64)
    state = (s.v, s.theta, s.i1, s.i2)
    half_dt = 0.5 * s.dt
    k1 = derivatives(s, state..., I_ext)
    k2 = derivatives(s, add_scaled(state, k1, half_dt)..., I_ext)
    k3 = derivatives(s, add_scaled(state, k2, half_dt)..., I_ext)
    k4 = derivatives(s, add_scaled(state, k3, s.dt)..., I_ext)
    candidate = ntuple(index -> state[index] + s.dt * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index]) / 6.0, 4)
    return candidate, finite_values(candidate...)
end

function step!(s::MihalasNieburNeuronState, I_ext::Float64=0.0)
    if !isfinite(I_ext) || !valid_runtime(s)
        return 0
    end
    candidate, ok = rk4_candidate(s, I_ext)
    if !ok
        return 0
    end
    s.v, s.theta, s.i1, s.i2 = candidate
    if s.v >= s.theta
        s.v = s.v_reset + s.b * (s.v - s.v_rest)
        s.theta = max(s.theta, s.theta_reset)
        s.i1 += s.r1
        s.i2 += s.r2
        return 1
    end
    return 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0)
    s = MihalasNieburNeuronState()
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

end # module MihalasNieburAccel
