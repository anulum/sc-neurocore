# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia retained normalized energy LIF

"""Retained normalized energy-gated exact-flow project recurrence."""
module SCNormalizedEnergyLifAccel

export SCNormalizedEnergyLIFNeuronState, candidate, simulate, step!, valid

"""Complete retained SC state and configuration."""
mutable struct SCNormalizedEnergyLIFNeuronState
    v::Float64; epsilon::Float64; v_rest::Float64; v_reset::Float64; v_threshold::Float64
    tau_m::Float64; tau_e::Float64; alpha::Float64; epsilon_0::Float64; resistance::Float64; dt::Float64
end

SCNormalizedEnergyLIFNeuronState() = SCNormalizedEnergyLIFNeuronState(-70.0, 1.0, -70.0, -70.0, -50.0, 10.0, 500.0, 0.1, 1.0, 1.0, 1.0)

"""Return the retained exact-flow candidate."""
function candidate(s::SCNormalizedEnergyLIFNeuronState, current::Float64)
    md = exp(-s.dt / s.tau_m); ed = exp(-s.dt / s.tau_e); de = s.epsilon - s.epsilon_0
    epsilon = s.epsilon_0 + de * ed; steady = s.epsilon_0 * s.tau_m * (1 - md)
    rate = 1 / s.tau_m - 1 / s.tau_e
    transient = abs(rate) < 1e-12 ? de * md * s.dt : de * md * expm1(rate * s.dt) / rate
    v = s.v_rest + (s.v - s.v_rest) * md + (s.resistance * current / s.tau_m) * (steady + transient)
    return v, epsilon
end

"""Return whether the retained state is valid."""
function valid(s::SCNormalizedEnergyLIFNeuronState)
    values = (s.v, s.epsilon, s.v_rest, s.v_reset, s.v_threshold, s.tau_m, s.tau_e,
              s.alpha, s.epsilon_0, s.resistance, s.dt)
    return all(isfinite, values) && -200 <= s.v <= 100 && -200 <= s.v_reset <= 100 &&
           0 <= s.epsilon <= s.epsilon_0 && s.tau_m > 0 && s.tau_e > 0 && s.alpha >= 0 &&
           s.epsilon_0 >= 0 && s.resistance > 0 && 0 < s.dt <= min(s.tau_m, s.tau_e) &&
           s.v_threshold > s.v_rest && s.v_threshold > s.v_reset
end

"""Advance one retained sample, returning `-1` atomically on failure."""
function step!(s::SCNormalizedEnergyLIFNeuronState, current::Float64=0.0)
    if !valid(s) || !isfinite(current); return -1; end
    v, epsilon = candidate(s, current)
    if !(isfinite(v) && -200 <= v <= 100 && isfinite(epsilon) && 0 <= epsilon <= s.epsilon_0); return -1; end
    if v >= s.v_threshold && epsilon > 0.1
        s.v, s.epsilon = s.v_reset, max(0.0, epsilon - s.alpha); return 1
    end
    s.v, s.epsilon = v, epsilon; return 0
end

"""Simulate a complete current trace from an explicitly supplied state."""
function simulate(currents::AbstractVector{<:Real}; state::SCNormalizedEnergyLIFNeuronState=SCNormalizedEnergyLIFNeuronState())
    voltages = zeros(length(currents)); epsilon = zeros(length(currents)); events = zeros(Int, length(currents))
    for i in eachindex(currents)
        events[i] = step!(state, Float64(currents[i]))
        voltages[i] = state.v
        epsilon[i] = state.epsilon
    end
    return (; voltages, epsilon, events, state)
end

end
