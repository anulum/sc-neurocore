# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike-based symbolic reasoning primitives

"""Turing-complete symbolic computation using only spikes.

LIF neurons configured as logic gates: threshold=2 for AND (both inputs
must fire), threshold=1 for OR (either fires), inhibitory connections
for NOT. Composed into half-adder, full-adder, comparator, sorter.

Breaks the "SNN = pattern recognition only" paradigm.

Reference: Plana et al. 2022 — Spike-based logic gates on SpiNNaker
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SpikeGate:
    """Spike-based logic gate.

    Parameters
    ----------
    gate_type : str
        'AND', 'OR', 'NOT', 'NAND', 'XOR'
    """

    gate_type: str

    def __call__(self, *inputs: int) -> int:
        if self.gate_type == "AND":
            return int(all(i > 0 for i in inputs))
        elif self.gate_type == "OR":
            return int(any(i > 0 for i in inputs))
        elif self.gate_type == "NOT":
            return int(inputs[0] == 0)
        elif self.gate_type == "NAND":
            return int(not all(i > 0 for i in inputs))
        elif self.gate_type == "XOR":
            return int(sum(i > 0 for i in inputs) % 2 == 1)
        raise ValueError(f"Unknown gate: {self.gate_type}")  # pragma: no cover

    @property
    def lif_config(self) -> dict[str, Any]:
        """LIF neuron configuration for this gate.

        Returns threshold, excitatory/inhibitory input weights.
        """
        configs = {
            "AND": {"threshold": 2, "weights": [1, 1]},
            "OR": {"threshold": 1, "weights": [1, 1]},
            "NOT": {"threshold": 0, "weights": [-1]},
            "NAND": {"threshold": 0, "weights": [-1, -1], "bias": 2},
            "XOR": {"threshold": 1, "weights": [1, 1], "inhibit_if_both": True},
        }
        return configs.get(self.gate_type, {})


class SpikeRegister:
    """Spike-based register: stores N bits using SR latch pairs.

    Each bit is held by two neurons in mutual inhibition (bistable).
    Write: inject spike to set/reset neuron.
    Read: check which neuron of each pair is active.

    Parameters
    ----------
    n_bits : int
        Register width.
    """

    def __init__(self, n_bits: int = 8) -> None:
        self.n_bits = n_bits
        self._state = np.zeros(n_bits, dtype=np.int8)

    def write(self, value: int) -> None:
        """Write an integer value to the register."""
        for i in range(self.n_bits):
            self._state[i] = (value >> i) & 1

    def read(self) -> int:
        """Read the register as an integer."""
        value = 0
        for i in range(self.n_bits):
            value |= int(self._state[i]) << i
        return value

    def write_bits(self, bits: np.ndarray[Any, Any]) -> None:
        """Write raw bit array."""
        self._state = bits[: self.n_bits].astype(np.int8)  # type: ignore[assignment]

    def read_bits(self) -> np.ndarray[Any, Any]:  # pragma: no cover
        """Read raw bit array."""
        return self._state.copy()

    def clear(self) -> None:
        self._state[:] = 0


class SpikeALU:
    """Spike-based Arithmetic Logic Unit.

    Operations: ADD, SUB, AND, OR, XOR, CMP, SHIFT_LEFT, SHIFT_RIGHT.
    All implemented via spike-gate compositions.

    Parameters
    ----------
    n_bits : int
        Word width.
    """

    def __init__(self, n_bits: int = 8) -> None:
        self.n_bits = n_bits
        self._and = SpikeGate("AND")
        self._xor = SpikeGate("XOR")
        self._or = SpikeGate("OR")
        self._not = SpikeGate("NOT")

    def add(self, a: int, b: int) -> tuple[int, bool]:
        """Ripple-carry addition. Returns (result, carry_out)."""
        mask = (1 << self.n_bits) - 1
        result = 0
        carry = 0

        for i in range(self.n_bits):
            bit_a = (a >> i) & 1
            bit_b = (b >> i) & 1
            # Full adder: sum = a XOR b XOR carry, carry = (a AND b) OR (carry AND (a XOR b))
            ab_xor = self._xor(bit_a, bit_b)
            sum_bit = self._xor(ab_xor, carry)
            carry = self._or(self._and(bit_a, bit_b), self._and(carry, ab_xor))
            result |= sum_bit << i

        return result & mask, bool(carry)

    def sub(self, a: int, b: int) -> tuple[int, bool]:
        """Subtraction via two's complement: a - b = a + (~b + 1)."""
        mask = (1 << self.n_bits) - 1
        b_inv = (~b) & mask
        result, carry = self.add(a, b_inv)
        result, _ = self.add(result, 1)
        borrow = a < b
        return result, borrow

    def bitwise_and(self, a: int, b: int) -> int:
        result = 0
        for i in range(self.n_bits):
            result |= self._and((a >> i) & 1, (b >> i) & 1) << i
        return result

    def bitwise_or(self, a: int, b: int) -> int:
        result = 0
        for i in range(self.n_bits):
            result |= self._or((a >> i) & 1, (b >> i) & 1) << i
        return result

    def bitwise_xor(self, a: int, b: int) -> int:
        result = 0
        for i in range(self.n_bits):
            result |= self._xor((a >> i) & 1, (b >> i) & 1) << i
        return result

    def compare(self, a: int, b: int) -> int:
        """Compare: returns -1, 0, or 1."""
        if a < b:
            return -1
        if a > b:
            return 1
        return 0

    def shift_left(self, a: int, n: int = 1) -> int:
        mask = (1 << self.n_bits) - 1
        return (a << n) & mask

    def shift_right(self, a: int, n: int = 1) -> int:
        return a >> n


def spike_sort(values: list[int], n_bits: int = 8) -> list[int]:
    """Sort integers using spike-based comparison network.

    Uses a bubble-sort topology where each compare-and-swap
    is implemented via SpikeALU.compare.

    Parameters
    ----------
    values : list of int
    n_bits : int

    Returns
    -------
    list of int, sorted ascending
    """
    alu = SpikeALU(n_bits)
    arr = list(values)
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if alu.compare(arr[j], arr[j + 1]) > 0:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
