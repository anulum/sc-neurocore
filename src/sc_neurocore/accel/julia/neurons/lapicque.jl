# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for lapicque

module LapicqueAccel

export step!, simulate, LapicqueNeuronState, valid, reset!

mutable struct LapicqueNeuronState
    v::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    tau::Float64
    resistance::Float64
    dt::Float64
end

function LapicqueNeuronState()
    LapicqueNeuronState(0.0, 0.0, 0.0, 1.0, 20.0, 1.0, 1.0)
end

function valid(s::LapicqueNeuronState)::Bool
    return isfinite(s.v) &&
        isfinite(s.v_rest) &&
        isfinite(s.v_reset) &&
        isfinite(s.v_threshold) && s.v_threshold > s.v_rest && s.v_threshold > s.v_reset &&
        s.v < s.v_threshold &&
        isfinite(s.tau) && s.tau > 0.0 &&
        isfinite(s.resistance) && s.resistance > 0.0 &&
        isfinite(s.dt) && s.dt > 0.0
end

function step!(s::LapicqueNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)::Int
    s.dt = dt
    if !isfinite(I_ext) || !valid(s)
        return 0
    end

    dv = (-(s.v - s.v_rest) + s.resistance * I_ext) / s.tau * s.dt
    next_v = s.v + dv
    if !isfinite(dv) || !isfinite(next_v)
        return 0
    end

    s.v = next_v
    if s.v >= s.v_threshold
        s.v = s.v_reset
        return 1
    end
    return 0
end

function reset!(s::LapicqueNeuronState)::Nothing
    s.v = s.v_rest
    return nothing
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=1.0)
    s = LapicqueNeuronState()
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext; dt=dt)
        trace[t] = s.v
        if result > 0
            spikes += 1
        end
    end
    return trace, spikes
end

end # module LapicqueAccel
