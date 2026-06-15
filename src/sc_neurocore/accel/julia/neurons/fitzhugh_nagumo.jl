# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for the FitzHugh-Nagumo RK4 simulator

# Parity contract: `simulate_trace` reproduces
# `sc_neurocore.neurons.models.fitzhugh_nagumo.FitzHughNagumoNeuron.simulate`
# bit-for-bit — the right-hand side is exact arithmetic (the cube is written
# `v*v*v`, matching Python/Rust/Go/Mojo; Julia's `v^3` for an integer literal
# also lowers to `v*v*v`), with no transcendental functions, and a 2-D flow
# cannot be chaotic, so identical operation order yields an identical trace.
#
# Reference: FitzHugh, R. (1961). Biophys. J. 1:445-466; Nagumo et al. (1962).

module FitzHughNagumoAccel

export simulate_trace

@inline function _rhs(v::Float64, w::Float64, a::Float64, b::Float64, eps::Float64, cur::Float64)
    dv = v - v * v * v / 3.0 - w + cur
    dw = eps * (v + a - b * w)
    return dv, dw
end

"""
    simulate_trace(v0, w0, a, b, epsilon, dt, v_threshold, n_steps, current)

Run `n_steps` RK4 updates of the FitzHugh-Nagumo system from state `(v0, w0)`
under a constant input `current`. Returns a named tuple `(trace, spikes, vf,
wf)` where `trace[t]` is `v` after step `t`, `spikes` counts upward crossings of
`v_threshold`, and `(vf, wf)` is the final state.
"""
function simulate_trace(
    v0::Float64,
    w0::Float64,
    a::Float64,
    b::Float64,
    epsilon::Float64,
    dt::Float64,
    v_threshold::Float64,
    n_steps::Int,
    current::Float64,
)
    trace = Vector{Float64}(undef, n_steps)
    v = v0
    w = w0
    spikes = 0
    for t in 1:n_steps
        v_prev = v
        k1v, k1w = _rhs(v, w, a, b, epsilon, current)
        k2v, k2w = _rhs(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w, a, b, epsilon, current)
        k3v, k3w = _rhs(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w, a, b, epsilon, current)
        k4v, k4w = _rhs(v + dt * k3v, w + dt * k3w, a, b, epsilon, current)
        v = v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        w = w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0
        trace[t] = v
        if v >= v_threshold && v_prev < v_threshold
            spikes += 1
        end
    end
    return (trace = trace, spikes = spikes, vf = v, wf = w)
end

end # module FitzHughNagumoAccel
