# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for the Izhikevich 2007 RK4 simulator

# Parity contract: `simulate_trace` reproduces
# `sc_neurocore.neurons.models.izhikevich2007.Izhikevich2007Neuron.simulate`
# bit-for-bit — the NeuroML right-hand side `k (v-vr)(v-vt)/C` is exact
# arithmetic (products, a sum and a division, no transcendental functions), so
# identical operation order yields an identical trace, spike count and final
# state.
#
# Reference: Izhikevich, E.M. (2007), Dynamical Systems in Neuroscience.

module Izhikevich2007Accel

export simulate_trace

@inline function _rhs(
    v::Float64, u::Float64, cap::Float64, k::Float64, vr::Float64, vt::Float64,
    a::Float64, b::Float64, cur::Float64,
)
    dv = (k * (v - vr) * (v - vt) - u + cur) / cap
    du = a * (b * (v - vr) - u)
    return dv, du
end

"""
    simulate_trace(v0, u0, cap, k, vr, vt, vpeak, a, b, c, d, dt, n_steps, current)

Run `n_steps` RK4 updates of the Izhikevich 2007 cell from state `(v0, u0)` under
a constant input `current`. Returns a named tuple `(trace, spikes, vf, uf)` where
`trace[t]` is `v` after step `t` (reset to `c` on spiking steps), `spikes` counts
the steps that reached `vpeak`, and `(vf, uf)` is the final state.
"""
function simulate_trace(
    v0::Float64,
    u0::Float64,
    cap::Float64,
    k::Float64,
    vr::Float64,
    vt::Float64,
    vpeak::Float64,
    a::Float64,
    b::Float64,
    c::Float64,
    d::Float64,
    dt::Float64,
    n_steps::Int,
    current::Float64,
)
    trace = Vector{Float64}(undef, n_steps)
    v = v0
    u = u0
    dt6 = dt / 6.0
    spikes = 0
    for t in 1:n_steps
        k1v, k1u = _rhs(v, u, cap, k, vr, vt, a, b, current)
        k2v, k2u = _rhs(v + 0.5 * dt * k1v, u + 0.5 * dt * k1u, cap, k, vr, vt, a, b, current)
        k3v, k3u = _rhs(v + 0.5 * dt * k2v, u + 0.5 * dt * k2u, cap, k, vr, vt, a, b, current)
        k4v, k4u = _rhs(v + dt * k3v, u + dt * k3u, cap, k, vr, vt, a, b, current)
        v = v + dt6 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        u = u + dt6 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        if v >= vpeak
            v = c
            u = u + d
            spikes += 1
        end
        trace[t] = v
    end
    return (trace = trace, spikes = spikes, vf = v, uf = u)
end

end # module Izhikevich2007Accel
