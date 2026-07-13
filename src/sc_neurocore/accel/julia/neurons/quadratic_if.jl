# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for quadratic_if

module QuadraticIfAccel

export step!, simulate, simulate_trace, QuadraticIFNeuronState, valid, reset!

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

function _exact_candidate(s::QuadraticIFNeuronState, I_ext::Float64)
    if I_ext > 0.0
        root_i = sqrt(I_ext)
        phase = atan(s.v / root_i)
        peak_phase = atan(s.v_peak / root_i)
        next_phase = phase + root_i * s.dt
        if next_phase >= peak_phase || next_phase >= pi / 2.0
            return s.v_reset, true
        end
        return root_i * tan(next_phase), false
    elseif I_ext == 0.0
        denominator = 1.0 - s.v * s.dt
        if denominator <= 0.0
            return s.v_reset, true
        end
        next_v = s.v / denominator
        return next_v >= s.v_peak ? (s.v_reset, true) : (next_v, false)
    end

    root_i = sqrt(-I_ext)
    if abs(s.v + root_i) <= 1.0e-15
        return s.v, false
    end
    numerator_ratio = (s.v - root_i) / (s.v + root_i)
    evolved_ratio = numerator_ratio * exp(2.0 * root_i * s.dt)
    denominator = 1.0 - evolved_ratio
    if (numerator_ratio < 1.0 && evolved_ratio >= 1.0) || abs(denominator) <= 1.0e-15
        return s.v_reset, true
    end
    next_v = root_i * (1.0 + evolved_ratio) / denominator
    return next_v >= s.v_peak ? (s.v_reset, true) : (next_v, false)
end

function step!(s::QuadraticIFNeuronState, I_ext::Float64=0.0; dt::Float64=s.dt)::Int
    if !all(isfinite, (s.v, s.v_reset, s.v_peak, dt, I_ext)) ||
       s.v >= s.v_peak ||
       s.v_reset >= s.v_peak ||
       dt <= 0.0
        throw(DomainError((s.v, I_ext), "QuadraticIF state/current must be finite and well-formed"))
    end

    candidate_state = QuadraticIFNeuronState(s.v, s.v_reset, s.v_peak, dt)
    next_v, spiked = _exact_candidate(candidate_state, I_ext)
    if !isfinite(next_v)
        throw(DomainError(next_v, "QuadraticIF exact-flow update became non-finite"))
    end

    s.dt = dt
    s.v = next_v
    if spiked
        return 1
    end
    return 0
end

function reset!(s::QuadraticIFNeuronState)::Nothing
    s.v = s.v_reset
    return nothing
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.01)
    result = simulate_trace(-1.0, -1.0, 1.0, dt, n_steps, I_ext)
    return result.trace, result.spikes
end

function simulate_trace(
    v::Float64,
    v_reset::Float64,
    v_peak::Float64,
    dt::Float64,
    n_steps::Int,
    I_ext::Float64,
)
    if n_steps < 0
        throw(ArgumentError("QuadraticIF n_steps must be non-negative"))
    end
    if !isfinite(I_ext)
        throw(DomainError(I_ext, "QuadraticIF input current must be finite"))
    end
    s = QuadraticIFNeuronState(v, v_reset, v_peak, dt)
    if !valid(s)
        throw(DomainError(v, "QuadraticIF state must satisfy the finite ordered contract"))
    end
    trace = zeros(n_steps)
    spikes = 0
    for t in 1:n_steps
        result = step!(s, I_ext)
        trace[t] = s.v
        if result > 0
            spikes += 1
        end
    end
    return (trace=trace, spikes=spikes, vf=s.v)
end

end # module QuadraticIfAccel
