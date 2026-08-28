# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia retained four-state GLIF recurrence

# Parity contract: `simulate_trace` reproduces
# `sc_neurocore.neurons.models.sc_four_state_glif.SCFourStateGLIFNeuron.simulate`. The retained four-state GLIF
# right-hand side is purely linear (no transcendental functions), so every RK4
# stage is exact arithmetic and the trace, spike count and final
# `(v, theta, i_asc1, i_asc2)` state are bit-identical to the NumPy reference.
#
# Project recurrence; no whole-model publication attribution.

module SCFourStateGLIFAccel

export simulate_trace

function simulate_trace(
    v0::Float64,
    theta0::Float64,
    theta_inf::Float64,
    i_asc1_0::Float64,
    i_asc2_0::Float64,
    v_rest::Float64,
    v_reset::Float64,
    tau_m::Float64,
    tau_theta::Float64,
    tau_asc1::Float64,
    tau_asc2::Float64,
    a_theta::Float64,
    delta_theta::Float64,
    r_asc1::Float64,
    r_asc2::Float64,
    resistance::Float64,
    dt::Float64,
    n_steps::Int,
    current::Float64,
)
    values = (
        v0, theta0, theta_inf, i_asc1_0, i_asc2_0, v_rest, v_reset,
        tau_m, tau_theta, tau_asc1, tau_asc2, a_theta, delta_theta,
        r_asc1, r_asc2, resistance, dt, current,
    )
    all(isfinite, values) || throw(ArgumentError("state, parameters and current must be finite"))
    all(>(0.0), (tau_m, tau_theta, tau_asc1, tau_asc2, dt)) ||
        throw(ArgumentError("time constants and dt must be positive"))
    delta_theta >= 0.0 || throw(ArgumentError("delta_theta must be non-negative"))
    resistance >= 0.0 || throw(ArgumentError("resistance must be non-negative"))
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative"))
    trace = Vector{Float64}(undef, n_steps)
    v = v0
    theta = theta0
    a1 = i_asc1_0
    a2 = i_asc2_0
    half_dt = 0.5 * dt
    deriv(vv, th, x1, x2) = (
        (-(vv - v_rest) + resistance * current + x1 + x2) / tau_m,
        (theta_inf - th + a_theta * (vv - v_rest)) / tau_theta,
        -x1 / tau_asc1,
        -x2 / tau_asc2,
    )
    spikes = 0
    @inbounds for t in 1:n_steps
        k1 = deriv(v, theta, a1, a2)
        k2 = deriv(
            v + half_dt * k1[1], theta + half_dt * k1[2],
            a1 + half_dt * k1[3], a2 + half_dt * k1[4],
        )
        k3 = deriv(
            v + half_dt * k2[1], theta + half_dt * k2[2],
            a1 + half_dt * k2[3], a2 + half_dt * k2[4],
        )
        k4 = deriv(
            v + dt * k3[1], theta + dt * k3[2],
            a1 + dt * k3[3], a2 + dt * k3[4],
        )
        v = v + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        theta = theta + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        a1 = a1 + dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0
        a2 = a2 + dt * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]) / 6.0
        all(isfinite, (v, theta, a1, a2)) ||
            throw(OverflowError("SC four-state GLIF candidate is non-finite"))
        if v >= theta
            v = v_reset
            theta += delta_theta
            a1 += r_asc1
            a2 += r_asc2
            spikes += 1
        end
        all(isfinite, (v, theta, a1, a2)) ||
            throw(OverflowError("SC four-state GLIF reset is non-finite"))
        trace[t] = v
    end
    return (trace = trace, spikes = spikes, vf = v, theta_f = theta, a1_f = a1, a2_f = a2)
end

end # module SCFourStateGLIFAccel
