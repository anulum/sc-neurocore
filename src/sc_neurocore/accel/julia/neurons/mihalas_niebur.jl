# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Mihalas-Niebur 2009 generalised IF (parity with mihalas_niebur.py)

# Parity contract: `simulate_trace` reproduces
# `sc_neurocore.neurons.models.mihalas_niebur.MihalasNieburNeuron.simulate`. The
# Mihalas-Niebur right-hand side is purely linear (no transcendental functions),
# so every RK4 stage is exact arithmetic and the trace, spike count and final
# `(v, theta, i1, i2)` state are bit-identical to the NumPy reference.
#
# Reference: Mihalas, S. & Niebur, E. (2009). Neural Comput. 21:704-718.

module MihalasNieburAccel

export simulate_trace

function simulate_trace(
    v0::Float64,
    theta0::Float64,
    i1_0::Float64,
    i2_0::Float64,
    v_rest::Float64,
    v_reset::Float64,
    theta_reset::Float64,
    theta_inf::Float64,
    tau_v::Float64,
    tau_theta::Float64,
    tau_1::Float64,
    tau_2::Float64,
    a::Float64,
    b::Float64,
    r1::Float64,
    r2::Float64,
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
    deriv(vv, th, j1, j2) = (
        (-(vv - v_rest) + j1 + j2 + current) / tau_v,
        (theta_inf - th + a * (vv - v_rest)) / tau_theta,
        -j1 / tau_1,
        -j2 / tau_2,
    )
    spikes = 0
    @inbounds for t in 1:n_steps
        k1 = deriv(v, theta, i1, i2)
        k2 = deriv(
            v + half_dt * k1[1], theta + half_dt * k1[2],
            i1 + half_dt * k1[3], i2 + half_dt * k1[4],
        )
        k3 = deriv(
            v + half_dt * k2[1], theta + half_dt * k2[2],
            i1 + half_dt * k2[3], i2 + half_dt * k2[4],
        )
        k4 = deriv(
            v + dt * k3[1], theta + dt * k3[2],
            i1 + dt * k3[3], i2 + dt * k3[4],
        )
        v = v + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        theta = theta + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        i1 = i1 + dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0
        i2 = i2 + dt * (k1[4] + 2.0 * k2[4] + 2.0 * k3[4] + k4[4]) / 6.0
        if v >= theta
            v = v_reset + b * (v - v_rest)
            theta = max(theta, theta_reset)
            i1 += r1
            i2 += r2
            spikes += 1
        end
        trace[t] = v
    end
    return (trace = trace, spikes = spikes, vf = v, theta_f = theta, i1_f = i1, i2_f = i2)
end

end # module MihalasNieburAccel
