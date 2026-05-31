# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for hindmarsh_rose

module HindmarshRoseAccel

export step!, simulate, HindmarshRoseNeuronState

mutable struct HindmarshRoseNeuronState
    x::Float64
    y::Float64
    z::Float64
    b::Float64
    r::Float64
    s::Float64
    x_rest::Float64
    dt::Float64
    x_threshold::Float64
end

function HindmarshRoseNeuronState()
    HindmarshRoseNeuronState(-1.6, -10.0, 2.0, 3.0, 0.001, 4.0, -1.6, 0.1, 1.0)
end

function _valid(s::HindmarshRoseNeuronState)
    values = (s.x, s.y, s.z, s.b, s.r, s.s, s.x_rest, s.dt, s.x_threshold)
    return all(isfinite, values) && s.r > 0.0 && s.s > 0.0 && s.dt > 0.0
end

function step!(s::HindmarshRoseNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)
    if !_valid(s) || !isfinite(I_ext) || !isfinite(dt) || dt <= 0.0
        return 0
    end
    x_prev = s.x
    x0, y0, z0 = s.x, s.y, s.z
    k1 = _derivatives(s, x0, y0, z0, I_ext)
    k1 === nothing && return 0
    k2 = _derivatives(s, x0 + 0.5 * dt * k1[1], y0 + 0.5 * dt * k1[2], z0 + 0.5 * dt * k1[3], I_ext)
    k2 === nothing && return 0
    k3 = _derivatives(s, x0 + 0.5 * dt * k2[1], y0 + 0.5 * dt * k2[2], z0 + 0.5 * dt * k2[3], I_ext)
    k3 === nothing && return 0
    k4 = _derivatives(s, x0 + dt * k3[1], y0 + dt * k3[2], z0 + dt * k3[3], I_ext)
    k4 === nothing && return 0
    next_x = x0 + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
    next_y = y0 + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
    next_z = z0 + (dt / 6.0) * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3])
    if !(isfinite(next_x) && isfinite(next_y) && isfinite(next_z))
        return 0
    end
    s.x = next_x
    s.y = next_y
    s.z = next_z
    return (s.x >= s.x_threshold && x_prev < s.x_threshold) ? 1 : 0
end

function _derivatives(s::HindmarshRoseNeuronState, x::Float64, y::Float64, z::Float64, I_ext::Float64)
    if !(isfinite(x) && isfinite(y) && isfinite(z) && isfinite(I_ext))
        return nothing
    end
    derivative = (
        y - x ^ 3 + s.b * x ^ 2 - z + I_ext,
        1.0 - 5.0 * x ^ 2 - y,
        s.r * (s.s * (x - s.x_rest) - z),
    )
    return all(isfinite, derivative) ? derivative : nothing
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = HindmarshRoseNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.x
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module HindmarshRoseAccel
