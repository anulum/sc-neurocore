# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — RISC-V deploy config generator (ported from tinysc_riscv/deploy.rs)

"""Generates .cargo/config.toml and memory.x linker scripts for RISC-V MCU targets."""

from __future__ import annotations

from sc_neurocore.edge.power_estimator import Board

_CARGO_CONFIGS = {
    Board.ESP32_C3: '[build]\ntarget = "riscv32imc-unknown-none-elf"\n\n'
    "[target.riscv32imc-unknown-none-elf]\n"
    'rustflags = ["-C", "link-arg=-Tmemory.x", "-C", "link-arg=-Tlink.x"]\n\n'
    '[unstable]\nbuild-std = ["core"]\n',
    Board.ESP32_C6: '[build]\ntarget = "riscv32imac-unknown-none-elf"\n\n'
    "[target.riscv32imac-unknown-none-elf]\n"
    'rustflags = ["-C", "link-arg=-Tmemory.x", "-C", "link-arg=-Tlink.x"]\n\n'
    '[unstable]\nbuild-std = ["core"]\n',
    Board.K210: '[build]\ntarget = "riscv64gc-unknown-none-elf"\n\n'
    "[target.riscv64gc-unknown-none-elf]\n"
    'rustflags = ["-C", "link-arg=-Tmemory.x", "-C", "link-arg=-Tlink.x"]\n\n'
    '[unstable]\nbuild-std = ["core"]\n',
}

_MEMORY_X = {
    Board.ESP32_C3: "MEMORY\n{\n    IRAM : ORIGIN = 0x40380000, LENGTH = 400K\n"
    "    DRAM : ORIGIN = 0x3FC80000, LENGTH = 400K\n"
    "    FLASH : ORIGIN = 0x42000000, LENGTH = 4M\n}\n",
    Board.GD32VF103: "MEMORY\n{\n    FLASH : ORIGIN = 0x08000000, LENGTH = 128K\n"
    "    RAM : ORIGIN = 0x20000000, LENGTH = 32K\n}\n",
}


def generate_cargo_config(board: Board) -> str:
    """Generate .cargo/config.toml content for a target board."""
    if board in _CARGO_CONFIGS:
        return _CARGO_CONFIGS[board]
    if board in (Board.ESP32_H2,):
        return _CARGO_CONFIGS[Board.ESP32_C3]
    return _CARGO_CONFIGS.get(
        Board.ESP32_C6,
        '[build]\ntarget = "riscv32imac-unknown-none-elf"\n\n'
        "[target.riscv32imac-unknown-none-elf]\n"
        'rustflags = ["-C", "link-arg=-Tmemory.x", "-C", "link-arg=-Tlink.x"]\n\n'
        '[unstable]\nbuild-std = ["core"]\n',
    )


def generate_memory_x(board: Board) -> str:
    """Generate memory.x linker script content for a target board."""
    if board in _MEMORY_X:
        return _MEMORY_X[board]
    if board in (Board.ESP32_C6, Board.ESP32_H2):
        return _MEMORY_X[Board.ESP32_C3]
    return (
        "MEMORY\n{\n    FLASH : ORIGIN = 0x08000000, LENGTH = 256K\n"
        "    RAM : ORIGIN = 0x20000000, LENGTH = 64K\n}\n"
    )
