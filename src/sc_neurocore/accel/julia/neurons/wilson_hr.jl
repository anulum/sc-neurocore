# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for wilson_hr

module WilsonHrAccel

export step!, simulate, WilsonHRNeuronState

mutable struct WilsonHRNeuronState
    v::Float64
    r::Float64
    tau_r::Float64
    v_peak::Float64
    dt::Float64
end

function WilsonHRNeuronState()
    WilsonHRNeuronState(-0.7, 0.1, 1.9, 0.4, 0.05)
end

function step!(s::WilsonHRNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    try
        poly = -(17.81 + 47.71 * s.v + 32.63 * s.v ^ 2) * (s.v - 0.55)
        syn = -26.0 * s.r * (s.v + 0.92)
        dv = (poly + syn + I_ext) * s.dt
        dr = (-s.r + 1.35 * s.v + 1.03) / s.tau_r * s.dt
        s.v += dv
        s.r += dr
        if s.v >= s.v_peak
            s.v = -0.7
            return 1
        end
        return 0
    catch _e
        return 0
    end
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = WilsonHRNeuronState()
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

end # module WilsonHrAccel
