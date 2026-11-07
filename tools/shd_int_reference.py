#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Integer reference of the SHD integer-delay network (bit-exact RTL contract)
"""Integer reference for the SHD integer-delay network (140 counts -> 128 -> 128 -> 20).

This module is the single definition of the hardware arithmetic. Every quantity is an integer;
there is no saturation, and every bound used to size the RTL is asserted on every sample.

Network, per sample with T input frames (T <= T_MAX):
  input   c[t, i]   counts per 2 ms frame, clamped to [0, COUNT_MAX]
  delay 1 y1[t, i] = c[t - d1[i], i]           for t in [0, T + 15), zero outside the input
  dense 1 a1[t, j] = sum_i q1[j, i] * y1[t, i]
  LIF 1   u = v - floor(v / 4) + a1 * 2^F ; spike if u >= theta1 ; v = 0 on spike else u
  delay 2 y2[t, j] = s1[t - d2[j], j]          for t in [0, T + 30), spikes of layer 1 masked to
                                               t < T + 15 as the PyTorch padding does
  dense 2 and LIF 2 as above with q2, theta2
  dense 3 a3[t, k] = sum_j q3[k, j] * s2[t, j]
  output  o[k]     = sum_t a3[t, k]             (argmax, lowest index on ties)
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

N_INPUT = 140
N_HIDDEN = 128
N_OUTPUT = 20
MAX_DELAY = 30  # delays 0..30, one 31-deep line per channel
DELAY_HALF = 15  # right padding of each PyTorch delay layer
T_MAX = 1024
COUNT_MAX = 15
WEIGHT_MAX = 127

L1_ACC_BOUND = N_INPUT * COUNT_MAX * WEIGHT_MAX  # 266 700
HIDDEN_ACC_BOUND = N_HIDDEN * WEIGHT_MAX  # 16 256
OUTPUT_BOUND = (T_MAX + 2 * DELAY_HALF) * HIDDEN_ACC_BOUND  # 17 133 824


class InputRefused(ValueError):
    """The input is outside the accepted range; the hardware refuses it the same way."""


@dataclass(frozen=True)
class IntNetwork:
    frac_bits: int
    d1: np.ndarray  # (140,) int, 0..30
    d2: np.ndarray  # (128,) int, 0..30
    q1: np.ndarray  # (128, 140) int, -127..127
    q2: np.ndarray  # (128, 128)
    q3: np.ndarray  # (20, 128)
    theta1: int
    theta2: int

    def __post_init__(self) -> None:
        for name, arr, shape in (
            ("d1", self.d1, (N_INPUT,)),
            ("d2", self.d2, (N_HIDDEN,)),
            ("q1", self.q1, (N_HIDDEN, N_INPUT)),
            ("q2", self.q2, (N_HIDDEN, N_HIDDEN)),
            ("q3", self.q3, (N_OUTPUT, N_HIDDEN)),
        ):
            if arr.shape != shape or arr.dtype.kind not in "iu":
                raise ValueError(
                    f"{name}: expected integer array of shape {shape}, got {arr.dtype} {arr.shape}"
                )
        for name, arr in (("d1", self.d1), ("d2", self.d2)):
            if arr.min() < 0 or arr.max() > MAX_DELAY:
                raise ValueError(f"{name} outside 0..{MAX_DELAY}")
        for name, arr in (("q1", self.q1), ("q2", self.q2), ("q3", self.q3)):
            if np.abs(arr).max() > WEIGHT_MAX:
                raise ValueError(f"{name} outside -{WEIGHT_MAX}..{WEIGHT_MAX}")
        if not (0 <= self.frac_bits <= 16) or self.theta1 < 1 or self.theta2 < 1:
            raise ValueError("frac_bits must be 0..16 and thresholds positive")

    def membrane_bound(self, layer: int) -> tuple[int, int]:
        """(lowest, highest) value the pre-compare membrane can take, from the invariant."""
        acc = L1_ACC_BOUND if layer == 1 else HIDDEN_ACC_BOUND
        theta = self.theta1 if layer == 1 else self.theta2
        x = acc << self.frac_bits
        return -4 * x, theta + x - 1

    def membrane_width(self, layer: int) -> int:
        lo, hi = self.membrane_bound(layer)
        return max((-lo).bit_length(), hi.bit_length()) + 1

    def to_json(self) -> str:
        return json.dumps(
            {
                "frac_bits": self.frac_bits,
                "theta1": self.theta1,
                "theta2": self.theta2,
                "d1": self.d1.tolist(),
                "d2": self.d2.tolist(),
                "q1": self.q1.tolist(),
                "q2": self.q2.tolist(),
                "q3": self.q3.tolist(),
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> "IntNetwork":
        raw = json.loads(text)
        return cls(
            frac_bits=int(raw["frac_bits"]),
            theta1=int(raw["theta1"]),
            theta2=int(raw["theta2"]),
            **{k: np.asarray(raw[k], dtype=np.int64) for k in ("d1", "d2", "q1", "q2", "q3")},
        )


def prepare_counts(counts: np.ndarray) -> np.ndarray:
    """Validate the frame matrix (T, 140) and clamp counts to COUNT_MAX."""
    c = np.asarray(counts)
    if c.ndim != 2 or c.shape[1] != N_INPUT:
        raise InputRefused(f"expected (T, {N_INPUT}) frames, got {c.shape}")
    if c.shape[0] < 1 or c.shape[0] > T_MAX:
        raise InputRefused(f"input length {c.shape[0]} outside 1..{T_MAX}")
    if np.any(c < 0) or np.any(c != np.floor(c)):
        raise InputRefused("counts must be non-negative integers")
    return np.minimum(c.astype(np.int64), COUNT_MAX)


def _delay(x: np.ndarray, d: np.ndarray, out_len: int) -> np.ndarray:
    """y[t, i] = x[t - d[i], i] for 0 <= t - d[i] < len(x), else 0."""
    y = np.zeros((out_len, x.shape[1]), dtype=np.int64)
    n = x.shape[0]
    for i, di in enumerate(d.tolist()):
        stop = min(n, out_len - di)
        if stop > 0:
            y[di : di + stop, i] = x[:stop, i]
    return y


def _matmul(x: np.ndarray, q: np.ndarray, bound: int) -> np.ndarray:
    # float64 BLAS is exact here: every partial sum is an integer below 2^53.
    acc = x.astype(np.float64) @ q.T.astype(np.float64)
    out = acc.astype(np.int64)
    assert np.array_equal(out, acc), "non-integer accumulator"
    assert np.abs(out).max(initial=0) <= bound, "accumulator bound violated"
    return out


def _lif(acc: np.ndarray, frac_bits: int, theta: int, bounds: tuple[int, int]) -> np.ndarray:
    lo, hi = bounds
    v = np.zeros(acc.shape[1], dtype=np.int64)
    spikes = np.zeros(acc.shape, dtype=np.int64)
    x_all = acc << frac_bits
    for t in range(acc.shape[0]):
        u = v - (v >> 2) + x_all[t]  # >> on int64 is the arithmetic shift: floor(v / 4)
        assert u.min() >= lo and u.max() <= hi, "membrane bound violated"
        fire = u >= theta
        spikes[t] = fire
        v = np.where(fire, 0, u)
    return spikes


def run(net: IntNetwork, counts: np.ndarray) -> np.ndarray:
    """The 20 output accumulators for one sample."""
    c = prepare_counts(counts)
    t_in = c.shape[0]
    t1 = t_in + DELAY_HALF
    t2 = t_in + 2 * DELAY_HALF
    a1 = _matmul(_delay(c, net.d1, t1), net.q1, L1_ACC_BOUND)
    s1 = _lif(a1, net.frac_bits, net.theta1, net.membrane_bound(1))
    a2 = _matmul(_delay(s1, net.d2, t2), net.q2, HIDDEN_ACC_BOUND)
    s2 = _lif(a2, net.frac_bits, net.theta2, net.membrane_bound(2))
    out = _matmul(s2, net.q3, HIDDEN_ACC_BOUND).sum(axis=0)
    assert np.abs(out).max(initial=0) <= OUTPUT_BOUND, "output bound violated"
    return out


def predict(net: IntNetwork, counts: np.ndarray) -> int:
    """Argmax of the output accumulators; ties go to the lowest index (numpy and torch agree)."""
    return int(np.argmax(run(net, counts)))
