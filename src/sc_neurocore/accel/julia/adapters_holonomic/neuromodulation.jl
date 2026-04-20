# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for adapters_holonomic/neuromodulation

module NeuromodulationAccel

using Statistics, LinearAlgebra

mutable struct NeuromodulatorSystemState
    da_level::Float64
    ht_level::Float64
    ne_level::Float64
end

function NeuromodulatorSystemState()
    NeuromodulatorSystemState(0.5, 0.5, 0.1)
end

function update_levels(s::NeuromodulatorSystemState, reward, stress)
    # Reward boosts Dopamine
    s.da_level += 0.1 * (reward - s.da_level)
    # Stress boosts Adrenaline (NE) && drops Serotonin (5HT)
    s.ne_level += 0.2 * (stress - s.ne_level)
    s.ht_level -= 0.1 * stress
    s.ht_level = clamp(s.ht_level, 0.1, 1.0)
end

function modulate_neuron(s::NeuromodulatorSystemState, neuron_params, Any])
    mod_params = neuron_params.copy()
    # Dopamine: Lowers Threshold (Excitation)
    if "v_threshold" in mod_params
        mod_params["v_threshold"] *= 1.0 - 0.2 * s.da_level
    # 5-HT reduces noise (stabilisation effect)
    if "noise_std" in mod_params
        mod_params["noise_std"] *= 1.0 - 0.5 * s.ht_level
    # Adrenaline: Increases Noise (Exploration) && Gain
    if "noise_std" in mod_params
        mod_params["noise_std"] += 0.1 * s.ne_level
    return mod_params
end

end # module NeuromodulationAccel
