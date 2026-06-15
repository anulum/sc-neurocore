# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for the FitzHugh-Rinzel RK4 simulator

# Parity contract: `simulate_trace` reproduces
# `sc_neurocore.neurons.models.fitzhugh_rinzel.FitzHughRinzelNeuron.simulate`
# bit-for-bit — the right-hand side is exact arithmetic (the cube is written
# `v*v*v`, matching Python/Rust/Go/Mojo; Julia's `v^3` for an integer literal
# also lowers to `v*v*v`), with no transcendental functions, so identical
# operation order yields an identical trace.
#
# Reference: FitzHugh, R. (1976); Rinzel, J. (1987).

module FitzHughRinzelAccel

export simulate_trace

@inline function _deriv(
    v::Float64, w::Float64, y::Float64, a::Float64, b::Float64, c::Float64, d::Float64,
    delta::Float64, mu::Float64, cur::Float64,
)
    dv = v - v * v * v / 3.0 - w + y + cur
    dw = delta * (a + v - b * w)
    dy = mu * (c - v - d * y)
    return dv, dw, dy
end

"""
    simulate_trace(v0, w0, y0, a, b, c, d, delta, mu, dt, v_threshold, n_steps, current)

Run `n_steps` RK4 updates of the FitzHugh-Rinzel system from state `(v0, w0, y0)`
under a constant input `current`. Returns a named tuple `(trace, spikes, vf, wf,
yf)` where `trace[t]` is `v` after step `t`, `spikes` counts upward crossings of
`v_threshold`, and `(vf, wf, yf)` is the final state.
"""
function simulate_trace(
    v0::Float64,
    w0::Float64,
    y0::Float64,
    a::Float64,
    b::Float64,
    c::Float64,
    d::Float64,
    delta::Float64,
    mu::Float64,
    dt::Float64,
    v_threshold::Float64,
    n_steps::Int,
    current::Float64,
)
    trace = Vector{Float64}(undef, n_steps)
    v = v0
    w = w0
    y = y0
    spikes = 0
    for t in 1:n_steps
        v_prev = v
        k1v, k1w, k1y = _deriv(v, w, y, a, b, c, d, delta, mu, current)
        k2v, k2w, k2y = _deriv(
            v + 0.5 * dt * k1v, w + 0.5 * dt * k1w, y + 0.5 * dt * k1y,
            a, b, c, d, delta, mu, current,
        )
        k3v, k3w, k3y = _deriv(
            v + 0.5 * dt * k2v, w + 0.5 * dt * k2w, y + 0.5 * dt * k2y,
            a, b, c, d, delta, mu, current,
        )
        k4v, k4w, k4y = _deriv(
            v + dt * k3v, w + dt * k3w, y + dt * k3y,
            a, b, c, d, delta, mu, current,
        )
        v = v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        w = w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0
        y = y + dt * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
        trace[t] = v
        if v >= v_threshold && v_prev < v_threshold
            spikes += 1
        end
    end
    return (trace = trace, spikes = spikes, vf = v, wf = w, yf = y)
end

end # module FitzHughRinzelAccel
