# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


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
    """

    data_width: int = 16
    fraction: int = 8
    v_rest: int = 0
    v_reset: int = 0
    v_threshold: int = 256  # 1.0 << 8
    refractory_period: int = 2

    def __post_init__(self) -> None:
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
        dv_leak = _mask((diff * leak_k) >> self.fraction, W)

        # --- Input term: I_t * gain_k >>> FRACTION ---
        dv_in = _mask((I_t * gain_k) >> self.fraction, W)

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
    """

    width: int = 16
    seed: int = 0xACE1

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
    """

    data_width: int = 16
    seed_init: int = 0xACE1

    def __post_init__(self) -> None:
        self.lfsr = FixedPointLFSR(width=self.data_width, seed=self.seed_init)

    def step(self, x_value: int) -> int:
        """Return 1 if LFSR < x_value, else 0 (one clock cycle)."""
        rnd = self.lfsr.reg
        self.lfsr.step()
        return 1 if rnd < x_value else 0

    def reset(self) -> None:
        self.lfsr.reset()
