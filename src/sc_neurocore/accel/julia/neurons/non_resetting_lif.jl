# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Source-faithful Kobayashi MAT(1) non-resetting neuron."""
module NonResettingLifAccel

export step!, simulate, NonResettingLIFNeuronState, valid, reset!, threshold

"""Complete MAT(1) state and documented numerical specialization."""
mutable struct NonResettingLIFNeuronState
    v::Float64
    theta::Float64
    refractory_remaining::Float64
    omega::Float64
    tau_m::Float64
    tau_theta::Float64
    alpha::Float64
    resistance::Float64
    refractory_period::Float64
    dt::Float64
end

"""Construct the enrolled MAT(1) specialization."""
function NonResettingLIFNeuronState()
    return NonResettingLIFNeuronState(0.0, 0.0, 0.0, 19.0, 5.0, 50.0, 37.0, 50.0, 2.0, 0.001)
end

"""Return the instantaneous adaptive threshold in millivolts."""
threshold(s::NonResettingLIFNeuronState)::Float64 = s.omega + s.theta

"""Return whether complete state and configuration satisfy the safety contract."""
function valid(s::NonResettingLIFNeuronState)::Bool
    return all(isfinite, (s.v, s.theta, s.refractory_remaining, s.omega, s.tau_m, s.tau_theta, s.alpha, s.resistance, s.refractory_period, s.dt)) &&
        -200.0 <= s.v <= 200.0 && 0.0 <= s.theta <= 1.0e9 && abs(s.omega) <= 1.0e9 &&
        0.0 <= s.alpha <= 1.0e9 && s.tau_m > 0.0 && s.tau_theta > 0.0 && s.resistance > 0.0 &&
        s.refractory_period >= 0.0 && s.dt > 0.0 && 0.0 <= s.refractory_remaining <= s.refractory_period
end

"""Advance one atomic source MAT(1) sample."""
function step!(s::NonResettingLIFNeuronState, current::Float64=0.0; dt::Float64=s.dt)::Int
    if !isfinite(current) || !isfinite(dt) || dt <= 0.0 || !valid(s)
        throw(DomainError((s.v, s.theta, current, dt), "invalid MAT(1) state, configuration, current, or timestep"))
    end
    next_v = s.v + dt * (-s.v + s.resistance * current) / s.tau_m
    next_theta = s.theta * exp(-dt / s.tau_theta)
    next_refractory = max(0.0, s.refractory_remaining - dt)
    if !all(isfinite, (next_v, next_theta, next_refractory)) || !(-200.0 <= next_v <= 200.0) || !(0.0 <= next_theta <= 1.0e9)
        throw(DomainError((next_v, next_theta, next_refractory), "MAT(1) candidate left the safety envelope"))
    end
    spike = next_refractory == 0.0 && next_v >= s.omega + next_theta
    if spike
        next_theta += s.alpha
        next_refractory = s.refractory_period
    end
    if !isfinite(next_theta) || next_theta > 1.0e9
        throw(DomainError(next_theta, "MAT(1) post-event threshold left the safety envelope"))
    end
    s.v = next_v
    s.theta = next_theta
    s.refractory_remaining = next_refractory
    s.dt = dt
    return spike ? 1 : 0
end

"""Restore zero-rest source state while retaining configuration."""
function reset!(s::NonResettingLIFNeuronState)::Nothing
    s.v = 0.0; s.theta = 0.0; s.refractory_remaining = 0.0
    return nothing
end

"""Simulate a configured MAT(1) current vector and return complete traces."""
function simulate(currents::AbstractVector{<:Real}; state::NonResettingLIFNeuronState=NonResettingLIFNeuronState())
    voltages = Vector{Float64}(undef, length(currents))
    thresholds = similar(voltages)
    refractory = similar(voltages)
    events = Vector{Int64}(undef, length(currents))
    for index in eachindex(currents)
        events[index] = step!(state, Float64(currents[index]); dt=state.dt)
        voltages[index] = state.v
        thresholds[index] = state.theta
        refractory[index] = state.refractory_remaining
    end
    return (; voltages, theta=thresholds, refractory, events, state)
end

"""Simulate a constant-current MAT(1) trace."""
function simulate(n_steps::Int=1000; current::Float64=0.7, dt::Float64=0.001)
    s = NonResettingLIFNeuronState(); trace = zeros(n_steps); spikes = 0
    for index in eachindex(trace)
        spikes += step!(s, current; dt=dt); trace[index] = s.v
    end
    return trace, spikes
end

end # module NonResettingLifAccel
