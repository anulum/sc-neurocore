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
    trace = Vector{Float64}(undef, n_steps)
    x = x0
    y = y0
    am1 = a * m1
    den = m0 + m1
    jmin = am1 / den
    jmax = (m0 + am1) / den
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
