# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC-Domain Spiking Neurons (ported from tinysc_riscv/neuron.rs)

"""LIF and Izhikevich spiking neurons operating in the SC domain.

Membrane potential is tracked as a popcount accumulator (integer, no FPU).
This mirrors the bare-metal implementation for RISC-V targets where
floating-point is unavailable or expensive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .bitstream import popcount_slice


@dataclass
class LifNeuron:
    """Leaky Integrate-and-Fire neuron (SC domain, integer arithmetic).

    Membrane potential = running popcount of input bitstream.
    Leak = right-shift per tick (exponential decay).
    Fires when potential exceeds threshold.
    """

    threshold: int = 512
    leak_shift: int = 3
    membrane: int = 0
    spike_count: int = 0

    def tick(self, input_words: list[int]) -> bool:
        """Process one timestep, return True if spike fired."""
        excitation = popcount_slice(input_words)
        self.membrane += excitation
        self.membrane -= self.membrane >> self.leak_shift
        if self.membrane >= self.threshold:
            self.membrane = 0
            self.spike_count += 1
            return True
        return False

    def reset(self) -> None:
        self.membrane = 0
        self.spike_count = 0


@dataclass
class IzhikevichNeuron:
    """Izhikevich neuron with integer SC-domain dynamics.

    Uses fixed-point arithmetic (Q16.16) to avoid floating-point.
    Supports regular spiking, fast spiking, chattering, and intrinsic burst.
    """

    a_q16: int = 1311  # 0.02 in Q16.16
    b_q16: int = 13107  # 0.2 in Q16.16
    c_q16: int = -4259840  # -65.0 in Q16.16
    d_q16: int = 524288  # 8.0 in Q16.16
    v_q16: int = -4259840  # -65.0
    u_q16: int = -917504  # -14.0

    spike_count: int = 0
    _q16_one: int = field(default=65536, repr=False)

    def tick(self, input_current_q16: int) -> bool:
        """Process one timestep. Returns True on spike."""
        v = self.v_q16
        u = self.u_q16

        dv = ((v * v) >> 14) + ((5 * v) >> 0) + (140 << 16) - u + input_current_q16
        du = (self.a_q16 * ((self.b_q16 * v >> 16) - u)) >> 16
        self.v_q16 = v + (dv >> 8)
        self.u_q16 = u + (du >> 8)

        if self.v_q16 >= (30 << 16):
            self.v_q16 = self.c_q16
            self.u_q16 += self.d_q16
            self.spike_count += 1
            return True
        return False

    def reset(self) -> None:
        self.v_q16 = self.c_q16
        self.u_q16 = -917504
        self.spike_count = 0

    @classmethod
    def regular_spiking(cls) -> IzhikevichNeuron:
        return cls(a_q16=1311, b_q16=13107, c_q16=-4259840, d_q16=524288)

    @classmethod
    def fast_spiking(cls) -> IzhikevichNeuron:
        return cls(a_q16=6554, b_q16=13107, c_q16=-4259840, d_q16=131072)

    @classmethod
    def chattering(cls) -> IzhikevichNeuron:
        return cls(a_q16=1311, b_q16=13107, c_q16=-3276800, d_q16=131072)

    @classmethod
    def intrinsic_burst(cls) -> IzhikevichNeuron:
        return cls(a_q16=1311, b_q16=13107, c_q16=-3604480, d_q16=262144)
