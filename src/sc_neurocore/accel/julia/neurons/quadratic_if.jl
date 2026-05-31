# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for quadratic_if

module QuadraticIfAccel

export step!, simulate, QuadraticIFNeuronState, valid, reset!

mutable struct QuadraticIFNeuronState
    v::Float64
    v_reset::Float64
    v_peak::Float64
    dt::Float64
end

function QuadraticIFNeuronState()
    QuadraticIFNeuronState(-1.0, -1.0, 1.0, 0.01)
end

function valid(s::QuadraticIFNeuronState)::Bool
    return all(isfinite, (s.v, s.v_reset, s.v_peak, s.dt)) &&
        s.v < s.v_peak &&
        s.v_reset < s.v_peak &&
        s.dt > 0.0
end

function step!(s::QuadraticIFNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)::Int
    if !isfinite(dt) || dt <= 0.0 || !isfinite(I_ext) || !valid(s)
        throw(DomainError((s.v, I_ext), "QuadraticIF state/current must be finite and well-formed"))
    end

    increment = (s.v * s.v + I_ext) * dt
    next_v = s.v + increment
    if !isfinite(increment) || !isfinite(next_v)
        throw(DomainError((increment, next_v), "QuadraticIF Euler update became non-finite"))
    end

    s.v = next_v
    if s.v >= s.v_peak
        s.v = s.v_reset
        return 1
    end
    return 0
end

function reset!(s::QuadraticIFNeuronState)::Nothing
    s.v = s.v_reset
    return nothing
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.01)
    s = QuadraticIFNeuronState()
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

end # module QuadraticIfAccel
