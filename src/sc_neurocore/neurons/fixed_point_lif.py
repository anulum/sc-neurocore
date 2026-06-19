# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mask to DATA_WIDTH bits and interpret as signed two's

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..constants import (
    FP_DATA_WIDTH,
    FP_FRACTION,
    FP_LFSR_SEED,
    FP_LFSR_WIDTH,
    FP_REFRACTORY_PERIOD,
    FP_V_THRESHOLD,
)


def _mask(value: int, width: int) -> int:
    """Mask to DATA_WIDTH bits and interpret as signed two's complement."""
    mask = (1 << width) - 1
    value = value & mask
    # Sign-extend: if MSB is set, value is negative
    if value >= (1 << (width - 1)):
        value -= 1 << width
    return value


@dataclass
class FixedPointLIFNeuron:
    """
    Bit-true fixed-point model of the Verilog ``sc_lif_neuron``.

    All arithmetic is performed in signed Q(FRACTION) fixed-point with
    explicit bit-width masking so that overflow/wrap behaviour matches
    the hardware exactly.

    Parameters
    ----------
    data_width : int
        Total bit width of all fixed-point values (default 16).
    fraction : int
        Number of fractional bits (default 8, giving Q8.8).
    v_rest, v_reset, v_threshold : int
        Membrane parameters in Q(FRACTION) fixed-point.
    refractory_period : int
        Number of clock cycles to hold after a spike.

    Example
    -------
    >>> neuron = FixedPointLIFNeuron()
    >>> spike, v = neuron.step(leak_k=240, gain_k=16, I_t=100)
    >>> spike in (0, 1)
    True
    >>> neuron.reset()
    """

    data_width: int = FP_DATA_WIDTH
    fraction: int = FP_FRACTION
    v_rest: int = 0
    v_reset: int = 0
    v_threshold: int = FP_V_THRESHOLD
    refractory_period: int = FP_REFRACTORY_PERIOD

    def __post_init__(self) -> None:
        if not 1 <= self.data_width <= 32:
            raise ValueError(f"data_width must be in [1, 32], got {self.data_width}")
        if not 0 <= self.fraction < self.data_width:
            raise ValueError(f"fraction must be in [0, data_width), got {self.fraction}")
        if self.refractory_period < 0:
            raise ValueError(f"refractory_period must be >= 0, got {self.refractory_period}")
        self.v: int = self.v_rest
        self.refractory_counter: int = 0

    def step(self, leak_k: int, gain_k: int, I_t: int, noise_in: int = 0) -> tuple[int, int]:
        """
        Execute one clock cycle — bit-true match to Verilog RTL.

        Parameters
        ----------
        leak_k : int   – ALPHA_LEAK in Q(FRACTION)
        gain_k : int   – GAIN_IN in Q(FRACTION)
        I_t    : int   – Input current in Q(FRACTION)
        noise_in : int – External noise in Q(FRACTION)

        Returns
        -------
        (spike, v_out) : tuple[int, int]
        """
        W = self.data_width

        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            self.v = self.v_rest
            return 0, _mask(self.v, W)

        # --- Leak term: (V_REST - v) * leak_k >>> FRACTION ---
        diff = _mask(self.v_rest - self.v, 2 * W)
        leak_mul = diff * leak_k
        # Arithmetic right shift (Python >> is arithmetic for negative ints)
        dv_leak = leak_mul >> self.fraction

        # --- Input term: I_t * gain_k >>> FRACTION ---
        in_mul = I_t * gain_k
        dv_in = in_mul >> self.fraction

        # --- Next membrane potential ---
        v_next = _mask(self.v + dv_leak + dv_in + noise_in, W)

        # --- Threshold check ---
        if v_next >= self.v_threshold:
            spike = 1
            self.v = self.v_reset
            self.refractory_counter = self.refractory_period
        else:
            spike = 0
            self.v = v_next

        return spike, _mask(self.v, W)

    def reset(self) -> None:
        """Reset neuron state to power-on defaults."""
        self.v = self.v_rest
        self.refractory_counter = 0

    # Aliases for BaseNeuron-compatible interface
    def reset_state(self) -> None:
        """Reset internal state (alias for :meth:`reset`)."""
        self.reset()

    def get_state(self) -> Dict[str, Any]:
        """Return dict with internal state."""
        return {
            "v": self.v,
            "refractory_counter": self.refractory_counter,
        }


@dataclass
class FixedPointLFSR:
    """
    Bit-true model of the 16-bit LFSR in ``sc_bitstream_encoder.v``.

    Polynomial: x^16 + x^14 + x^13 + x^11 + 1
    Taps (0-indexed): 15, 13, 12, 10

    Example
    -------
    >>> lfsr = FixedPointLFSR(seed=0xACE1)
    >>> vals = [lfsr.step() for _ in range(10)]
    >>> len(set(vals)) > 1  # produces varying pseudo-random values
    True
    """

    width: int = FP_LFSR_WIDTH
    seed: int = FP_LFSR_SEED

    def __post_init__(self) -> None:
        if self.seed == 0:
            raise ValueError("LFSR seed must be non-zero.")
        self.reg: int = self.seed & ((1 << self.width) - 1)

    def step(self) -> int:
        """Advance one clock cycle; return new register state."""
        w = self.width
        feedback = (
            ((self.reg >> (w - 1)) & 1)
            ^ ((self.reg >> (w - 3)) & 1)
            ^ ((self.reg >> (w - 4)) & 1)
            ^ ((self.reg >> (w - 6)) & 1)
        )
        self.reg = ((self.reg << 1) & ((1 << w) - 1)) | feedback
        return self.reg

    def reset(self, seed: int | None = None) -> None:
        self.reg = (seed if seed is not None else self.seed) & ((1 << self.width) - 1)


@dataclass
class FixedPointBitstreamEncoder:
    """
    Bit-true model of ``sc_bitstream_encoder.v``.

    Combines LFSR + comparator to produce a stochastic bitstream
    where P(bit=1) ~ x_value / (2^DATA_WIDTH - 1).

    Example
    -------
    >>> enc = FixedPointBitstreamEncoder(seed_init=0xACE1)
    >>> bits = [enc.step(x_value=128) for _ in range(100)]
    >>> all(b in (0, 1) for b in bits)
    True
    """

    data_width: int = FP_DATA_WIDTH
    seed_init: int = FP_LFSR_SEED

    def __post_init__(self) -> None:
        self.lfsr = FixedPointLFSR(width=self.data_width, seed=self.seed_init)

    def step(self, x_value: int) -> int:
        """Return 1 if LFSR < x_value, else 0 (one clock cycle)."""
        rnd = self.lfsr.reg
        self.lfsr.step()
        return 1 if rnd < x_value else 0

    def reset(self) -> None:
        self.lfsr.reset()
