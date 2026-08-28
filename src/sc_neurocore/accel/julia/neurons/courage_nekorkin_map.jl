# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Courbage-Nekorkin-Vdovin 2007 map (parity with courage_nekorkin_map.py)

# Parity contract: `simulate_trace` reproduces
# `sc_neurocore.neurons.models.courage_nekorkin_map.CourageNekorkinMapNeuron.simulate`
# bit-for-bit. The map is exact floating-point arithmetic (additions,
# multiplications, one division for the breakpoints, and a piecewise/Heaviside
# branch — no transcendental functions), so an identical operation order yields
# an identical trace, upward-crossing spike count, and final state.
#
# Reference: Courbage, M., Nekorkin, V.I. & Vdovin, L.V. (2007).
# Chaos 17:043109 (arXiv:0712.2097), eqs. 3-5.

module CourageNekorkinMapAccel

export simulate_trace

function simulate_trace(
    x0::Float64,
    y0::Float64,
    m0::Float64,
    m1::Float64,
    a::Float64,
    d::Float64,
    j::Float64,
    beta::Float64,
    eps::Float64,
    x_threshold::Float64,
    n_steps::Int,
    current::Float64,
)
    values = (x0, y0, m0, m1, a, d, j, beta, eps, x_threshold, current)
    all(isfinite, values) || throw(ArgumentError("Courbage inputs must be finite"))
    n_steps >= 0 || throw(ArgumentError("n_steps must be non-negative"))
    0.0 < m0 < 1.0 || throw(ArgumentError("m0 must satisfy 0 < m0 < 1"))
    m1 > 0.0 || throw(ArgumentError("m1 must be positive"))
    0.0 < a < 1.0 || throw(ArgumentError("a must satisfy 0 < a < 1"))
    d > 0.0 && beta > 0.0 && eps > 0.0 || throw(ArgumentError("d, beta, and eps must be positive"))
    0.0 < j < d || throw(ArgumentError("J must satisfy 0 < J < d"))
    trace = Vector{Float64}(undef, n_steps)
    x = x0
    y = y0
    am1 = a * m1
    den = m0 + m1
    jmin = am1 / den
    jmax = (m0 + am1) / den
    jmin < d < jmax || throw(ArgumentError("d must satisfy Jmin < d < Jmax"))
    spikes = 0
    @inbounds for t in 1:n_steps
        x_prev = x
        if x <= jmin
            fx = -m0 * x
        elseif x < jmax
            fx = m1 * (x - a)
        else
            fx = -m0 * (x - 1.0)
        end
        h = (x - d) >= 0.0 ? 1.0 : 0.0
        x_new = x + fx - y - beta * h + current
        y_new = y + eps * (x - j)
        isfinite(x_new) && isfinite(y_new) || throw(OverflowError("Courbage candidate became non-finite"))
        x = x_new
        y = y_new
        trace[t] = x
        if x >= x_threshold && x_prev < x_threshold
            spikes += 1
        end
    end
    return (trace = trace, spikes = spikes, xf = x, yf = y)
end

end # module CourageNekorkinMapAccel
