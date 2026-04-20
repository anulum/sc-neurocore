# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for terman_wang

module TermanWangAccel

export step!, simulate, TermanWangOscillatorState

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

function step!(s::TermanWangOscillatorState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        f = 3.0 * s.v - s.v ^ 3 + 2.0
        g = s.alpha * (1.0 + tanh(s.v / s.beta))
        dv = (f - s.w + I_ext + s.rho) * s.dt
        dw = s.epsilon * (g - s.w) * s.dt
        v_prev = s.v
        s.v += dv
        s.w += dw
        return (s.v >= s.v_peak && v_prev < s.v_peak) ? 1 : 0
    catch _e
        return 0
    end
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
