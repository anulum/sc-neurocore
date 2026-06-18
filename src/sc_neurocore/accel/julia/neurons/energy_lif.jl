# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia for energy_lif

module EnergyLifAccel

export step!, simulate, valid, exact_candidate, EnergyLIFNeuronState

const V_MIN = -200.0
const V_MAX = 100.0
const ENERGY_GATE = 0.1

mutable struct EnergyLIFNeuronState
    v::Float64
    epsilon::Float64
    v_rest::Float64
    v_reset::Float64
    v_threshold::Float64
    tau_m::Float64
    tau_e::Float64
    alpha::Float64
    epsilon_0::Float64
    resistance::Float64
    dt::Float64
end

function EnergyLIFNeuronState()
    EnergyLIFNeuronState(-70.0, 1.0, -70.0, -70.0, -50.0, 10.0, 500.0, 0.1, 1.0, 1.0, 1.0)
end

function step!(s::EnergyLIFNeuronState, I_ext::Float64=0.0; dt::Float64=0.1)
    _ = dt
    if !valid(s) || !isfinite(I_ext)
        return -1
    end
    v_candidate, epsilon_candidate = exact_candidate(s, I_ext)
    if !(isfinite(v_candidate) && V_MIN <= v_candidate <= V_MAX && isfinite(epsilon_candidate) && 0.0 <= epsilon_candidate <= s.epsilon_0)
        return -1
    end
    if v_candidate >= s.v_threshold && epsilon_candidate > ENERGY_GATE
        epsilon_after_spike = max(0.0, epsilon_candidate - s.alpha)
        if !(isfinite(epsilon_after_spike) && epsilon_after_spike <= s.epsilon_0)
            return -1
        end
        s.v = s.v_reset
        s.epsilon = epsilon_after_spike
        return 1
    end
    s.v = v_candidate
    s.epsilon = epsilon_candidate
    return 0
end

function exact_candidate(s::EnergyLIFNeuronState, I_ext::Float64)
    membrane_decay = exp(-s.dt / s.tau_m)
    energy_decay = exp(-s.dt / s.tau_e)
    energy_delta = s.epsilon - s.epsilon_0
    epsilon_candidate = s.epsilon_0 + energy_delta * energy_decay
    steady_energy_integral = s.epsilon_0 * s.tau_m * (1.0 - membrane_decay)
    coupled_rate = (1.0 / s.tau_m) - (1.0 / s.tau_e)
    transient_energy_integral = if abs(coupled_rate) < 1.0e-12
        energy_delta * membrane_decay * s.dt
    else
        energy_delta * membrane_decay * expm1(coupled_rate * s.dt) / coupled_rate
    end
    v_candidate = s.v_rest + (s.v - s.v_rest) * membrane_decay +
                  (s.resistance * I_ext / s.tau_m) * (steady_energy_integral + transient_energy_integral)
    return v_candidate, epsilon_candidate
end

function valid(s::EnergyLIFNeuronState)
    return isfinite(s.v) &&
        V_MIN <= s.v <= V_MAX &&
        isfinite(s.epsilon) &&
        s.epsilon >= 0.0 &&
        isfinite(s.v_rest) &&
        isfinite(s.v_reset) &&
        V_MIN <= s.v_reset <= V_MAX &&
        isfinite(s.v_threshold) &&
        isfinite(s.tau_m) &&
        s.tau_m > 0.0 &&
        isfinite(s.tau_e) &&
        s.tau_e > 0.0 &&
        isfinite(s.alpha) &&
        s.alpha >= 0.0 &&
        isfinite(s.epsilon_0) &&
        s.epsilon_0 >= 0.0 &&
        isfinite(s.resistance) &&
        s.resistance > 0.0 &&
        isfinite(s.dt) &&
        s.dt > 0.0 &&
        s.epsilon <= s.epsilon_0 &&
        s.dt <= s.tau_m &&
        s.dt <= s.tau_e &&
        s.v_threshold > s.v_rest &&
        s.v_threshold > s.v_reset
end

function simulate(n_steps::Int=1000; I_ext::Float64=10.0, dt::Float64=0.1)
    s = EnergyLIFNeuronState()
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

end # module EnergyLifAccel
