# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Pernarowski 1994 beta-cell burster (parity with pernarowski.py)

# Parity contract: `simulate_trace` reproduces
# `sc_neurocore.neurons.models.pernarowski.PernarowskiNeuron.simulate` bit-for-bit.
# The right-hand side is exact polynomial arithmetic (the cubic uses `v*v*v`,
# matching the engine `v.powi(3)`; no transcendental functions), so an identical
# RK4 operation order yields an identical `v` trace, upward-crossing spike count,
# and final `(v, w, z)` state.
#
# Reference: Pernarowski, M. (1994). SIAM J. Appl. Math. 54:814-832.

module PernarowskiAccel

export simulate_trace

function simulate_trace(
    v0::Float64,
    w0::Float64,
    z0::Float64,
    alpha::Float64,
    beta::Float64,
    eps1::Float64,
    eps2::Float64,
    gamma::Float64,
    dt::Float64,
    v_threshold::Float64,
    n_steps::Int,
    current::Float64,
)
    trace = Vector{Float64}(undef, n_steps)
    v = v0
    w = w0
    z = z0
    deriv(vv, ww, zz) = (
        vv - vv * vv * vv / 3.0 - ww - zz + current,
        eps1 * (vv - gamma * ww + alpha),
        eps2 * (beta * (vv + 0.7) - zz),
    )
    spikes = 0
    @inbounds for t in 1:n_steps
        v_prev = v
        dv1, dw1, dz1 = deriv(v, w, z)
        dv2, dw2, dz2 = deriv(v + 0.5 * dt * dv1, w + 0.5 * dt * dw1, z + 0.5 * dt * dz1)
        dv3, dw3, dz3 = deriv(v + 0.5 * dt * dv2, w + 0.5 * dt * dw2, z + 0.5 * dt * dz2)
        dv4, dw4, dz4 = deriv(v + dt * dv3, w + dt * dw3, z + dt * dz3)
        v = v + dt * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4) / 6.0
        w = w + dt * (dw1 + 2.0 * dw2 + 2.0 * dw3 + dw4) / 6.0
        z = z + dt * (dz1 + 2.0 * dz2 + 2.0 * dz3 + dz4) / 6.0
        trace[t] = v
        if v >= v_threshold && v_prev < v_threshold
            spikes += 1
        end
    end
    return (trace = trace, spikes = spikes, vf = v, wf = w, zf = z)
end

end # module PernarowskiAccel
