# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for galves_locherbach

module GalvesLocherbachAccel

export step!, simulate, validate_galves_locherbach, GalvesLocherbachNeuronState

mutable struct GalvesLocherbachNeuronState
    v::Float64
    v_rest::Float64
    decay::Float64
    threshold_rate::Float64
    steepness::Float64
    dt::Float64
end

function GalvesLocherbachNeuronState()
    GalvesLocherbachNeuronState(0.0, 0.0, 0.95, 0.5, 5.0, 1.0)
end

function _firing_prob(s::GalvesLocherbachNeuronState)
    z = s.steepness * (s.v - s.threshold_rate)
    if z >= 0.0
        tail = exp(-z)
        return 1.0 / (1.0 + tail)
    end
    tail = exp(z)
    return tail / (1.0 + tail)
end

function step!(s::GalvesLocherbachNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    if !validate_galves_locherbach(s) || !isfinite(I_ext)
        return 0
    end
    s.v = s.decay * s.v + I_ext
    p = _firing_prob(s) * s.dt
    spike = rand() < p ? 1 : 0
    if spike == 1
        s.v = s.v_rest
    end
    return spike
end

function validate_galves_locherbach(s::GalvesLocherbachNeuronState)
    return isfinite(s.v) && isfinite(s.v_rest) && isfinite(s.threshold_rate) &&
           isfinite(s.decay) && 0.0 <= s.decay <= 1.0 &&
           isfinite(s.steepness) && s.steepness > 0.0 &&
           isfinite(s.dt) && 0.0 < s.dt <= 1.0
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = GalvesLocherbachNeuronState()
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

end # module GalvesLocherbachAccel
