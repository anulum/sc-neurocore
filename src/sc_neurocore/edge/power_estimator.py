# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Edge power estimation (ported from tinysc_riscv/power.rs)

"""Power consumption and memory footprint estimation for RISC-V MCU targets.

Enables pre-deployment validation that a network fits in target board
RAM/flash and provides µW power estimates at given clock frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Board(Enum):
    """Supported RISC-V MCU targets."""

    ESP32_C3 = ("ESP32-C3", 400, 4096, 15_000, 5)
    ESP32_C6 = ("ESP32-C6", 512, 4096, 18_000, 7)
    ESP32_H2 = ("ESP32-H2", 320, 4096, 12_000, 3)
    GD32VF103 = ("GD32VF103", 32, 128, 8_000, 10)
    CH32V307 = ("CH32V307", 64, 256, 10_000, 8)
    K210 = ("K210", 8192, 16384, 300_000, 50)
    GENERIC = ("Generic", 64, 256, 10_000, 10)

    def __init__(self, label: str, ram_kb: int, flash_kb: int, active_uw: int, sleep_uw: int):
        self.label = label
        self.ram_kb = ram_kb
        self.flash_kb = flash_kb
        self._active_uw_ref = active_uw
        self._sleep_uw = sleep_uw


@dataclass
class PowerProfile:
    """Estimated power profile for a target board at a given clock."""

    board: Board
    clock_mhz: int
    active_uw: int
    sleep_uw: int

    @classmethod
    def for_board(cls, board: Board, clock_mhz: int = 160) -> PowerProfile:
        scaled = board._active_uw_ref * clock_mhz // 160
        return cls(board=board, clock_mhz=clock_mhz, active_uw=scaled, sleep_uw=board._sleep_uw)

    def duty_cycled_uw(self, duty: float) -> int:
        """Estimate µW for a given duty cycle (0.0=sleep, 1.0=active)."""
        return int(self.active_uw * duty + self.sleep_uw * (1.0 - duty))


@dataclass
class MemoryFootprint:
    """Memory footprint estimate for a tinySC network."""

    stack_bytes: int
    static_bytes: int
    total_bytes: int
    fits_in_ram: bool
    fits_in_flash: bool

    @classmethod
    def estimate(
        cls, num_layers: int, neurons_per_layer: int, bs_words: int, board: Board
    ) -> MemoryFootprint:
        """Estimate memory for a network configuration.

        Parameters
        ----------
        num_layers : int
            Number of layers.
        neurons_per_layer : int
            Max neurons in any layer.
        bs_words : int
            Bitstream words per neuron.
        board : Board
            Target board.
        """
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

    @staticmethod
    def max_neurons(board: Board) -> int:
        """Maximum neurons that fit in a board's RAM (single layer)."""
        ram = board.ram_kb * 1024
        overhead = 512
        if ram <= overhead:
            return 0
        return (ram - overhead) // 12
