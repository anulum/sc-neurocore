# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia Fardet-Levina eLIF

"""Coupled RK4 author-Brian implementation of Fardet-Levina eLIF."""
module EnergyLifAccel

export EnergyLIFNeuronState, candidate, simulate, step!, valid

"""Complete source eLIF state and configuration."""
mutable struct EnergyLIFNeuronState
    v::Float64; epsilon::Float64; capacitance::Float64; g_leak::Float64
    e_0::Float64; e_u::Float64; e_d::Float64; e_f::Float64
    v_threshold::Float64; v_reset::Float64; alpha::Float64; epsilon_0::Float64
    epsilon_c::Float64; delta::Float64; tau_e::Float64; dt::Float64
end

EnergyLIFNeuronState() = EnergyLIFNeuronState(-61.0, 0.32, 100.0, 9.0, -62.5, -58.5, -40.0, -62.0, -59.0, -62.0, 1.0, 0.5, 0.18, 0.01, 200.0, 0.1)

function rhs(s::EnergyLIFNeuronState, v::Float64, epsilon::Float64, current::Float64)
    leak = s.e_0 + (s.e_u - s.e_0) * (1.0 - epsilon / s.epsilon_0)
    dv = (s.g_leak * (leak - v) + current) / s.capacitance
    de = ((1.0 - epsilon / (s.alpha * s.epsilon_0))^3 - (v - s.e_f) / (s.e_d - s.e_f)) / s.tau_e
    return dv, de
end

"""Return one simultaneous source RK4 candidate."""
function candidate(s::EnergyLIFNeuronState, current::Float64)
    dt = s.dt
    k1 = rhs(s, s.v, s.epsilon, current)
    k2 = rhs(s, s.v + dt * k1[1] / 2, s.epsilon + dt * k1[2] / 2, current)
    k3 = rhs(s, s.v + dt * k2[1] / 2, s.epsilon + dt * k2[2] / 2, current)
    k4 = rhs(s, s.v + dt * k3[1], s.epsilon + dt * k3[2], current)
    return s.v + dt * (k1[1] + 2k2[1] + 2k3[1] + k4[1]) / 6,
           s.epsilon + dt * (k1[2] + 2k2[2] + 2k3[2] + k4[2]) / 6
end

"""Return whether the complete source state is valid."""
function valid(s::EnergyLIFNeuronState)
    values = (s.v, s.epsilon, s.capacitance, s.g_leak, s.e_0, s.e_u, s.e_d, s.e_f,
              s.v_threshold, s.v_reset, s.alpha, s.epsilon_0, s.epsilon_c, s.delta, s.tau_e, s.dt)
    return all(isfinite, values) && -200 <= s.v <= 100 && -200 <= s.v_reset <= 100 &&
           0 <= s.epsilon <= 5 && s.capacitance > 0 && s.g_leak > 0 && s.alpha > 0 &&
           s.epsilon_0 > 0 && s.epsilon_c >= 0 && s.delta >= 0 && s.tau_e > 0 &&
           0 < s.dt <= min(1.0, s.tau_e) && s.e_d != s.e_f && s.v_threshold > s.v_reset
end

"""Advance one source sample, returning `-1` without mutation on failure."""
function step!(s::EnergyLIFNeuronState, current::Float64=0.0)
    if !valid(s) || !isfinite(current); return -1; end
    v, epsilon = candidate(s, current)
    if !(isfinite(v) && -200 <= v <= 100 && isfinite(epsilon) && 0 <= epsilon <= 5); return -1; end
    if v > s.v_threshold && epsilon > s.epsilon_c
        after = epsilon - s.delta
        if !(0 <= after <= 5); return -1; end
        s.v, s.epsilon = s.v_reset, after
        return 1
    end
    s.v, s.epsilon = v, epsilon
    return 0
end

"""Simulate a complete current trace from an explicitly supplied state."""
function simulate(currents::AbstractVector{<:Real}; state::EnergyLIFNeuronState=EnergyLIFNeuronState())
    voltages = zeros(length(currents)); epsilon = zeros(length(currents)); events = zeros(Int, length(currents))
    for i in eachindex(currents)
        events[i] = step!(state, Float64(currents[i]))
        voltages[i] = state.v
        epsilon[i] = state.epsilon
    end
    return (; voltages, epsilon, events, state)
end

end
