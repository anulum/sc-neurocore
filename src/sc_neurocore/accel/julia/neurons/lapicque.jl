# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for lapicque

module LapicqueAccel

export step!, simulate, simulate_trace, LapicqueNeuronState, valid, reset!

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
    if !isfinite(I_ext)
        throw(DomainError(I_ext, "Lapicque input current must be finite"))
    end
    if !isfinite(dt) || dt <= 0.0
        throw(DomainError(dt, "Lapicque dt must be finite and positive"))
    end
    if !valid(s)
        throw(DomainError(s.v, "Lapicque state must satisfy finite positive-RC threshold contract"))
    end

    v_inf = s.v_rest + s.resistance * I_ext
    decay = exp(-dt / s.tau)
    next_v = v_inf + (s.v - v_inf) * decay
    if !isfinite(v_inf) || !isfinite(decay) || !isfinite(next_v)
        throw(DomainError(next_v, "Lapicque voltage candidate must remain finite"))
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
    result = simulate_trace(0.0, 0.0, 0.0, 1.0, 20.0, 1.0, dt, n_steps, I_ext)
    return result.trace, result.spikes
end

function simulate_trace(
    v::Float64,
    v_rest::Float64,
    v_reset::Float64,
    v_threshold::Float64,
    tau::Float64,
    resistance::Float64,
    dt::Float64,
    n_steps::Int,
    I_ext::Float64,
)
    if n_steps < 0
        throw(ArgumentError("Lapicque n_steps must be non-negative"))
    end
    if !isfinite(I_ext)
        throw(DomainError(I_ext, "Lapicque input current must be finite"))
    end
    s = LapicqueNeuronState(v, v_rest, v_reset, v_threshold, tau, resistance, dt)
    if !valid(s)
        throw(DomainError(v, "Lapicque state must satisfy finite positive-RC threshold contract"))
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

end # module LapicqueAccel
