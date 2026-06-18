# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for mat

module MatAccel

export step!, simulate, MATNeuronState, rk4_candidate

const V_MIN = -200.0
const V_MAX = 100.0
const THETA_MAX = 1.0e9

mutable struct MATNeuronState
    v::Float64
    theta1::Float64
    theta2::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold_base::Float64
    tau_m::Float64
    tau_1::Float64
    tau_2::Float64
    h1::Float64
    h2::Float64
    resistance::Float64
    dt::Float64
end

function MATNeuronState()
    MATNeuronState(-70.0, 0.0, 0.0, -70.0, -70.0, -50.0, 10.0, 10.0, 200.0, 5.0, 3.0, 1.0, 1.0)
end

finite(x::Float64)::Bool = isfinite(x)

function valid_state(s::MATNeuronState)::Bool
    finite(s.v) &&
        finite(s.theta1) &&
        finite(s.theta2) &&
        finite(s.v_rest) &&
        finite(s.v_reset) &&
        finite(s.v_threshold_base) &&
        finite(s.tau_m) &&
        finite(s.tau_1) &&
        finite(s.tau_2) &&
        finite(s.h1) &&
        finite(s.h2) &&
        finite(s.resistance) &&
        finite(s.dt) &&
        V_MIN <= s.v <= V_MAX &&
        V_MIN <= s.v_reset <= V_MAX &&
        0.0 <= s.theta1 <= THETA_MAX &&
        0.0 <= s.theta2 <= THETA_MAX &&
        0.0 <= s.h1 <= THETA_MAX &&
        0.0 <= s.h2 <= THETA_MAX &&
        s.tau_m > 0.0 &&
        s.tau_1 > 0.0 &&
        s.tau_2 > 0.0 &&
        s.resistance > 0.0 &&
        s.dt > 0.0
end

function derivatives(s::MATNeuronState, v::Float64, theta1::Float64, theta2::Float64, I_ext::Float64)
    dv = (-(v - s.v_rest) + s.resistance * I_ext) / s.tau_m
    return dv, -theta1 / s.tau_1, -theta2 / s.tau_2
end

function rk4_candidate(s::MATNeuronState, I_ext::Float64)
    k1v, k1t1, k1t2 = derivatives(s, s.v, s.theta1, s.theta2, I_ext)
    k2v, k2t1, k2t2 = derivatives(
        s,
        s.v + 0.5 * s.dt * k1v,
        s.theta1 + 0.5 * s.dt * k1t1,
        s.theta2 + 0.5 * s.dt * k1t2,
        I_ext,
    )
    k3v, k3t1, k3t2 = derivatives(
        s,
        s.v + 0.5 * s.dt * k2v,
        s.theta1 + 0.5 * s.dt * k2t1,
        s.theta2 + 0.5 * s.dt * k2t2,
        I_ext,
    )
    k4v, k4t1, k4t2 = derivatives(
        s,
        s.v + s.dt * k3v,
        s.theta1 + s.dt * k3t1,
        s.theta2 + s.dt * k3t2,
        I_ext,
    )
    scale = s.dt / 6.0
    return (
        s.v + scale * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
        s.theta1 + scale * (k1t1 + 2.0 * k2t1 + 2.0 * k3t1 + k4t1),
        s.theta2 + scale * (k1t2 + 2.0 * k2t2 + 2.0 * k3t2 + k4t2),
    )
end

function step!(s::MATNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    _ = dt
    if !finite(I_ext) || !valid_state(s)
        return -1
    end
    v_candidate, theta1_candidate, theta2_candidate = rk4_candidate(s, I_ext)
    if !(finite(v_candidate) && finite(theta1_candidate) && finite(theta2_candidate))
        return -1
    end
    if !(V_MIN <= v_candidate <= V_MAX && 0.0 <= theta1_candidate <= THETA_MAX && 0.0 <= theta2_candidate <= THETA_MAX)
        return -1
    end
    threshold = s.v_threshold_base + theta1_candidate + theta2_candidate
    if v_candidate >= threshold
        theta1_after_spike = theta1_candidate + s.h1
        theta2_after_spike = theta2_candidate + s.h2
        if !(finite(theta1_after_spike) && finite(theta2_after_spike) && theta1_after_spike <= THETA_MAX && theta2_after_spike <= THETA_MAX)
            return -1
        end
        s.v = s.v_reset
        s.theta1 = theta1_after_spike
        s.theta2 = theta2_after_spike
        return 1
    end
    s.v = v_candidate
    s.theta1 = theta1_candidate
    s.theta2 = theta2_candidate
    return 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = MATNeuronState()
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

end # module MatAccel
