# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for profiling/energy

module EnergyAccel

using Statistics, LinearAlgebra

mutable struct EnergyMetricsState
    E_AND::Float64
    E_XOR::Float64
    E_ADD::Float64
    E_MEM::Float64
    total_ops_and::Float64
    total_ops_xor::Float64
    total_bits_mem::Float64
end

function EnergyMetricsState()
    EnergyMetricsState(1e-16, 1.5e-16, 5e-16, 5e-15, 0.0, 0.0, 0.0)
end

function reset(s::EnergyMetricsState)
    s.total_ops_and = 0
    s.total_ops_xor = 0
    s.total_bits_mem = 0
end

function estimate_energy(s::EnergyMetricsState)
    e_logic = (s.total_ops_and * s.E_AND) + (s.total_ops_xor * s.E_XOR)
    e_mem = s.total_bits_mem * s.E_MEM
    return e_logic + e_mem
end

function co2_emission_g(s::EnergyMetricsState, carbon_intensity_g_per_kwh)
    # Energy in Joules -> kWh -> Grams CO2
    # 1 J = 2.77e-7 kWh
    kwh = s.estimate_energy() * 2.7778e-7
    return kwh * carbon_intensity_g_per_kwh
end

function track_energy(func)
    res = func(*args, ^kwargs)
    # Determine 'self' object
    # 1. If func is a bound method, it has __self__
    layer_obj = getattr(func, "__self__", nothing)
    # 2. If used on class def, args[0] is self
    if layer_obj is nothing && length(args) > 0
        # Check if args[0] looks like a layer
        if hasattr(args[0], "n_neurons")
            layer_obj = args[0]
    if (
        layer_obj
        && hasattr(layer_obj, "n_neurons")
        && hasattr(layer_obj, "n_inputs")
        && hasattr(layer_obj, "length")
    )
        # Dense Layer Ops
        ops = layer_obj.n_inputs * layer_obj.n_neurons * layer_obj.length
        profiler.total_ops_and += ops
        # Memory Read
        mem = (layer_obj.n_neurons * layer_obj.n_inputs * layer_obj.length) + (
            layer_obj.n_inputs * layer_obj.length
        )
        profiler.total_bits_mem += mem
    return res
    return wrapper
end

end # module EnergyAccel
