# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia source-faithful Mihalas-Niebur batch kernel

"""
Mihalaş-Niebur equations 2.1–2.2 with fixed-grid RK4 and sampled events.

Rates are per millisecond, voltages are volts, and currents are divided by
capacitance. The event map is exactly `Iⱼ ← RⱼIⱼ + Aⱼ`, `V ← Vᵣ`, and
`Θ ← max(Θᵣ, Θ)`.
"""
module MihalasNieburAccel

export simulate_trace

"""Return voltage trace, event count, and final four-state tuple."""
function simulate_trace(
    v0::Float64,
    theta0::Float64,
    i1_0::Float64,
    i2_0::Float64,
    v_rest::Float64,
    v_reset::Float64,
    theta_reset::Float64,
    theta_inf::Float64,
    leak_rate::Float64,
    threshold_voltage_coupling::Float64,
    threshold_decay_rate::Float64,
    current_decay_rate_1::Float64,
    current_decay_rate_2::Float64,
    current_retention_1::Float64,
    current_retention_2::Float64,
    current_jump_1::Float64,
    current_jump_2::Float64,
    dt::Float64,
    n_steps::Int,
    current::Float64,
)
    trace = Vector{Float64}(undef, n_steps)
    v = v0
    theta = theta0
    i1 = i1_0
    i2 = i2_0
    half_dt = 0.5 * dt
    derivatives(voltage, threshold, current_1, current_2) = (
        current + current_1 + current_2 - leak_rate * (voltage - v_rest),
        threshold_voltage_coupling * (voltage - v_rest) -
        threshold_decay_rate * (threshold - theta_inf),
        -current_decay_rate_1 * current_1,
        -current_decay_rate_2 * current_2,
    )
    events = 0
    @inbounds for index in 1:n_steps
        k1 = derivatives(v, theta, i1, i2)
        k2 = derivatives(
            v + half_dt * k1[1], theta + half_dt * k1[2],
            i1 + half_dt * k1[3], i2 + half_dt * k1[4],
        )
        k3 = derivatives(
            v + half_dt * k2[1], theta + half_dt * k2[2],
            i1 + half_dt * k2[3], i2 + half_dt * k2[4],
        )
        k4 = derivatives(
            v + dt * k3[1], theta + dt * k3[2],
            i1 + dt * k3[3], i2 + dt * k3[4],
        )
        v += dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        theta += dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        i1 += dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0
        i2 += dt * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]) / 6.0
        if v >= theta
            i1 = current_retention_1 * i1 + current_jump_1
            i2 = current_retention_2 * i2 + current_jump_2
            v = v_reset
            theta = max(theta_reset, theta)
            events += 1
        end
        trace[index] = v
    end
    return (trace = trace, spikes = events, vf = v, theta_f = theta, i1_f = i1, i2_f = i2)
end

end # module MihalasNieburAccel
