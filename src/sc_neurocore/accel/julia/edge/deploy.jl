# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia acceleration for edge/deploy

module DeployAccel

using Statistics, LinearAlgebra

function generate_cargo_config(board)
    if board in _CARGO_CONFIGS
        return _CARGO_CONFIGS[board]
    if board in (Board.ESP32_H2,)
        return _CARGO_CONFIGS[Board.ESP32_C3]
    return _CARGO_CONFIGS.get(Board.ESP32_C6,
        '[build]\ntarget = "riscv32imac-unknown-none-elf"\n\n'
        '[target.riscv32imac-unknown-none-elf]\n'
        'rustflags = ["-C", "link-arg=-Tmemory.x", "-C", "link-arg=-Tlink.x"]\n\n'
        '[unstable]\nbuild-std = ["core"]\n')
end

function generate_memory_x(board)
    if board in _MEMORY_X
        return _MEMORY_X[board]
    if board in (Board.ESP32_C6, Board.ESP32_H2)
        return _MEMORY_X[Board.ESP32_C3]
    return ('MEMORY\n{\n    FLASH : ORIGIN = 0x08000000, LENGTH = 256K\n'
            '    RAM : ORIGIN = 0x20000000, LENGTH = 64K\n}\n')
end

end # module DeployAccel
