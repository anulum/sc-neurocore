# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for edge/power_estimator

module PowerEstimatorAccel

using Statistics, LinearAlgebra

mutable struct MemoryFootprintState
    label::Float64
    ram_kb::Float64
    flash_kb::Float64
    _active_uw_ref::Float64
    _sleep_uw::Float64
    board::Float64
    clock_mhz::Float64
    active_uw::Float64
    sleep_uw::Float64
    stack_bytes::Float64
    static_bytes::Float64
    total_bytes::Float64
    fits_in_ram::Float64
    fits_in_flash::Float64
end

function MemoryFootprintState()
    MemoryFootprintState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
end

function for_board(s::MemoryFootprintState)
    scaled = board._active_uw_ref * clock_mhz // 160
    return cls(board=board, clock_mhz=clock_mhz,
               active_uw=scaled, sleep_uw=board._sleep_uw)
end

function duty_cycled_uw(s::MemoryFootprintState, duty)
    return int(s.active_uw * duty + s.sleep_uw * (1.0 - duty))
end

function estimate(s::MemoryFootprintState)
    bs_words: int, board: Board) -> MemoryFootprint
    neuron_size = 12
    layer_size = neuron_size * neurons_per_layer + 32
    net_size = layer_size * num_layers + 16
    bs_stack = bs_words * 4
    stack = net_size + bs_stack + 256
    static_code = 8192
    total = stack + static_code
    ram_bytes = board.ram_kb * 1024
    flash_bytes = board.flash_kb * 1024
    return cls(
    stack_bytes=stack,
    static_bytes=static_code,
    total_bytes=total,
    fits_in_ram=stack <= ram_bytes,
    fits_in_flash=static_code <= flash_bytes,
    )
end

function max_neurons(s::MemoryFootprintState)
    ram = board.ram_kb * 1024
    overhead = 512
    if ram <= overhead
        return 0
    return (ram - overhead) // 12
end

end # module PowerEstimatorAccel
