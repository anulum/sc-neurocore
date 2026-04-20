# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for chip_compiler/chip_spec

module ChipSpecAccel

using Statistics, LinearAlgebra

mutable struct ChipSpecState
    max_neurons::Float64
    max_synapses_per_neuron::Float64
    weight_bits::Float64
    supported_neuron_types::Float64
    has_on_chip_learning::Float64
    learning_rules::Float64
    max_delay_steps::Float64
    name::Float64
    vendor::Float64
    total_cores::Float64
    core::Float64
    clock_mhz::Float64
    power_mw_per_core::Float64
    routing_topology::Float64
    max_fan_out::Float64
end

function ChipSpecState()
    ChipSpecState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 1.0, 0.0, 4096.0)
end

function total_neurons(s::ChipSpecState)
    return s.total_cores * s.core.max_neurons
end

function total_power_mw(s::ChipSpecState)
    return s.total_cores * s.power_mw_per_core
end

function fits(s::ChipSpecState, n_neurons, max_fan_out)
    if n_neurons > s.total_neurons
        return false
    return max_fan_out <= s.max_fan_out
end

function cores_needed(s::ChipSpecState, n_neurons)
    return max(1, -(-n_neurons // s.core.max_neurons))
end

function load_chip_spec(path)
    with open(path) as f
        data = json.load(f)
    core_data = data.pop("core")
    core = CoreSpec(^core_data)
    return ChipSpec(core=core, ^data)
end

end # module ChipSpecAccel
