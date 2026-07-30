# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Retained SC non-resetting exact-relaxation adaptive LIF recurrence."""
module SCNonResettingAdaptiveLifAccel

export step!, simulate, SCNonResettingAdaptiveLIFNeuronState, valid, reset!

"""Complete historical project state and configuration."""
mutable struct SCNonResettingAdaptiveLIFNeuronState
    v::Float64; theta::Float64; v_rest::Float64; theta_rest::Float64
    delta_theta::Float64; tau_m::Float64; tau_theta::Float64; r_m::Float64; dt::Float64
end

"""Construct the frozen project defaults."""
SCNonResettingAdaptiveLIFNeuronState() = SCNonResettingAdaptiveLIFNeuronState(-65.0, -50.0, -65.0, -50.0, 5.0, 10.0, 50.0, 1.0, 0.1)

"""Return whether complete project state and configuration are valid."""
function valid(s::SCNonResettingAdaptiveLIFNeuronState)::Bool
    return all(isfinite, (s.v, s.theta, s.v_rest, s.theta_rest, s.delta_theta, s.tau_m, s.tau_theta, s.r_m, s.dt)) && s.delta_theta >= 0.0 && s.r_m >= 0.0 && s.tau_m > 0.0 && s.tau_theta > 0.0 && s.dt > 0.0
end

"""Advance one atomic exact-relaxation project sample."""
function step!(s::SCNonResettingAdaptiveLIFNeuronState, current::Float64=0.0; dt::Float64=s.dt)::Int
    if !isfinite(current) || !isfinite(dt) || dt <= 0.0 || !valid(s)
        throw(DomainError((s.v, s.theta, current, dt), "invalid SC adaptive LIF state, configuration, current, or timestep"))
    end
    steady = s.v_rest + s.r_m * current
    dv = exp(-dt / s.tau_m); dtheta = exp(-dt / s.tau_theta)
    next_v = dv * s.v + (1.0 - dv) * steady
    next_theta = dtheta * s.theta + (1.0 - dtheta) * s.theta_rest
    if !all(isfinite, (steady, next_v, next_theta))
        throw(DomainError((steady, next_v, next_theta), "SC adaptive LIF candidate became non-finite"))
    end
    spike = next_v >= next_theta
    if spike; next_theta += s.delta_theta; end
    if !isfinite(next_theta); throw(DomainError(next_theta, "SC adaptive LIF threshold became non-finite")); end
    s.v = next_v; s.theta = next_theta; s.dt = dt
    return spike ? 1 : 0
end

"""Restore voltage and threshold to configured rests."""
function reset!(s::SCNonResettingAdaptiveLIFNeuronState)::Nothing
    s.v = s.v_rest; s.theta = s.theta_rest; return nothing
end

"""Simulate a configured project current vector and return complete traces."""
function simulate(currents::AbstractVector{<:Real}; state::SCNonResettingAdaptiveLIFNeuronState=SCNonResettingAdaptiveLIFNeuronState())
    voltages = Vector{Float64}(undef, length(currents))
    thresholds = similar(voltages)
    events = Vector{Int64}(undef, length(currents))
    for index in eachindex(currents)
        events[index] = step!(state, Float64(currents[index]); dt=state.dt)
        voltages[index] = state.v
        thresholds[index] = state.theta
    end
    return (; voltages, theta=thresholds, events, state)
end

"""Simulate a constant-current retained-project trace."""
function simulate(n_steps::Int=1000; current::Float64=20.0, dt::Float64=0.1)
    s = SCNonResettingAdaptiveLIFNeuronState(); trace = zeros(n_steps); spikes = 0
    for index in eachindex(trace)
        spikes += step!(s, current; dt=dt); trace[index] = s.v
    end
    return trace, spikes
end

end # module SCNonResettingAdaptiveLifAccel
