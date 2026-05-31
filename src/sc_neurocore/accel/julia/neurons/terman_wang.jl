# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for terman_wang

module TermanWangAccel

export step!, simulate, validate, TermanWangOscillatorState

mutable struct TermanWangOscillatorState
    v::Float64
    w::Float64
    alpha::Float64
    beta::Float64
    epsilon::Float64
    rho::Float64
    dt::Float64
    v_peak::Float64
end

function TermanWangOscillatorState()
    TermanWangOscillatorState(-1.5, -0.5, 3.0, 0.2, 0.02, 0.0, 0.05, 1.5)
end

function validate(s::TermanWangOscillatorState, dt::Float64=s.dt)::Bool
    return isfinite(s.v) &&
        isfinite(s.w) &&
        isfinite(s.alpha) &&
        isfinite(s.beta) &&
        s.beta > 0.0 &&
        isfinite(s.epsilon) &&
        s.epsilon > 0.0 &&
        isfinite(s.rho) &&
        isfinite(dt) &&
        dt > 0.0 &&
        isfinite(s.v_peak)
end

function step!(s::TermanWangOscillatorState, I_ext::Float64=0.0; dt::Float64=s.dt)
    if !validate(s, dt) || !isfinite(I_ext)
        return -1
    end

    f = 3.0 * s.v - s.v ^ 3 + 2.0
    g = s.alpha * (1.0 + tanh(s.v / s.beta))
    dv = (f - s.w + I_ext + s.rho) * dt
    dw = s.epsilon * (g - s.w) * dt
    next_v = s.v + dv
    next_w = s.w + dw
    if !isfinite(dv) || !isfinite(dw) || !isfinite(next_v) || !isfinite(next_w)
        return -1
    end

    v_prev = s.v
    s.v = next_v
    s.w = next_w
    return (s.v >= s.v_peak && v_prev < s.v_peak) ? 1 : 0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = TermanWangOscillatorState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v
        if result isa Number && result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module TermanWangAccel
