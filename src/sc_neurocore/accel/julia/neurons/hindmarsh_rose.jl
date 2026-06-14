# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for the Hindmarsh-Rose RK4 simulator

# Parity contract: `simulate_trace` reproduces
# `sc_neurocore.neurons.models.hindmarsh_rose.HindmarshRoseNeuron.simulate`
# bit-for-bit — the right-hand side is exact arithmetic (the square and cube are
# written `x*x` and `(x*x)*x`, matching Python/Rust/Go/Mojo), with no
# transcendental functions, so identical operation order yields an identical
# trace even though the bursting dynamics are chaotic.
#
# Reference: Hindmarsh, J.L. & Rose, R.M. (1984). Proc. R. Soc. Lond. B 221:87-102.

module HindmarshRoseAccel

export simulate_trace

@inline function _deriv(
    x::Float64, y::Float64, z::Float64, b::Float64, r::Float64, s::Float64,
    x_rest::Float64, cur::Float64,
)
    x2 = x * x
    x3 = x2 * x
    dx = y - x3 + b * x2 - z + cur
    dy = 1.0 - 5.0 * x2 - y
    dz = r * (s * (x - x_rest) - z)
    return dx, dy, dz
end

"""
    simulate_trace(x0, y0, z0, b, r, s, x_rest, dt, x_threshold, n_steps, current)

Run `n_steps` RK4 updates of the Hindmarsh-Rose system from state `(x0, y0, z0)`
under a constant input `current`. Returns a named tuple `(trace, spikes, xf, yf,
zf)` where `trace[t]` is `x` after step `t`, `spikes` counts upward crossings of
`x_threshold`, and `(xf, yf, zf)` is the final state.
"""
function simulate_trace(
    x0::Float64,
    y0::Float64,
    z0::Float64,
    b::Float64,
    r::Float64,
    s::Float64,
    x_rest::Float64,
    dt::Float64,
    x_threshold::Float64,
    n_steps::Int,
    current::Float64,
)
    trace = Vector{Float64}(undef, n_steps)
    x = x0
    y = y0
    z = z0
    dt6 = dt / 6.0
    spikes = 0
    for t in 1:n_steps
        x_prev = x
        k1x, k1y, k1z = _deriv(x, y, z, b, r, s, x_rest, current)
        k2x, k2y, k2z = _deriv(
            x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z,
            b, r, s, x_rest, current,
        )
        k3x, k3y, k3z = _deriv(
            x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z,
            b, r, s, x_rest, current,
        )
        k4x, k4y, k4z = _deriv(
            x + dt * k3x, y + dt * k3y, z + dt * k3z,
            b, r, s, x_rest, current,
        )
        x = x + dt6 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        y = y + dt6 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
        z = z + dt6 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
        trace[t] = x
        if x >= x_threshold && x_prev < x_threshold
            spikes += 1
        end
    end
    return (trace = trace, spikes = spikes, xf = x, yf = y, zf = z)
end

end # module HindmarshRoseAccel
