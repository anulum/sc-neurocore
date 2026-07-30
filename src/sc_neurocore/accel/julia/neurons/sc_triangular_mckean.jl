# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — retained triangular McKean-like RK4 backend

# Parity contract: `simulate_trace` reproduces
# `sc_neurocore.neurons.models.sc_triangular_mckean.SCTriangularMcKeanNeuron.simulate` bit-for-bit. The
# piecewise-linear right-hand side is exact arithmetic (additions, multiplications
# and branch selection — no transcendental functions), so an identical RK4
# operation order yields an identical `v` trace, upward-crossing spike count, and
# final `(v, w)` state.
#
# Reference: SC project recurrence; no external paper attribution.

module SCTriangularMcKeanAccel

export simulate_trace

"""Execute the retained SC recurrence under a constant drive."""
function simulate_trace(
    v0::Float64,
    w0::Float64,
    a::Float64,
    eps::Float64,
    gamma::Float64,
    dt::Float64,
    v_peak::Float64,
    n_steps::Int,
    current::Float64,
)
    trace = Vector{Float64}(undef, n_steps)
    v = v0
    w = w0
    half_a = a / 2.0
    mid = (1.0 + a) / 2.0
    fv(x) = x < half_a ? -x : (x < mid ? x - a : 1.0 - x)
    spikes = 0
    @inbounds for t in 1:n_steps
        v_prev = v
        dv1 = fv(v) - w + current
        dw1 = eps * (v - gamma * w)
        v2 = v + 0.5 * dt * dv1
        w2 = w + 0.5 * dt * dw1
        dv2 = fv(v2) - w2 + current
        dw2 = eps * (v2 - gamma * w2)
        v3 = v + 0.5 * dt * dv2
        w3 = w + 0.5 * dt * dw2
        dv3 = fv(v3) - w3 + current
        dw3 = eps * (v3 - gamma * w3)
        v4 = v + dt * dv3
        w4 = w + dt * dw3
        dv4 = fv(v4) - w4 + current
        dw4 = eps * (v4 - gamma * w4)
        v = v + dt * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4) / 6.0
        w = w + dt * (dw1 + 2.0 * dw2 + 2.0 * dw3 + dw4) / 6.0
        trace[t] = v
        if v >= v_peak && v_prev < v_peak
            spikes += 1
        end
    end
    return (trace = trace, spikes = spikes, vf = v, wf = w)
end

end # module SCTriangularMcKeanAccel
