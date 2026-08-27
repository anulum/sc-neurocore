# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC resetting windowed neuron (preserved repository recurrence)

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SCResettingParallelSpikingNeuron:
    """SC resetting windowed neuron — preserved repository recurrence.

    Historical repository model formerly published under the
    ``ParallelSpikingNeuron`` name. It is structurally distinct from the
    Fang et al. (2023) sliding PSN in two ways: the retained-input
    buffer is zeroed whenever the neuron fires (the PSN family has no
    reset), and a replaced ``kernel`` is dotted against circular buffer
    slots rather than time-ordered inputs, so for non-uniform kernels
    the weight-to-input pairing rotates with the write pointer. During
    warm-up the score divides by the full ``kernel_size``, which for
    the default uniform kernel matches zero-padded pre-history.

    score[t] = kernel[:n] . buffer[:n],  n = min(t+1, kernel_size)
    spike when score >= v_threshold, then buffer[:] = 0.

    Count-neutral SC identity: it consumes no source-catalogue slot and
    makes no publication-exact claim. Finite-input trajectories are
    preserved bit-for-bit from the pre-2026-08-27 implementation.
    """

    kernel_size: int = 8
    v_threshold: float = 1.0
    kernel: np.ndarray[Any, Any] = field(init=False)
    buffer: np.ndarray[Any, Any] = field(init=False)
    _ptr: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not isinstance(self.kernel_size, int) or self.kernel_size < 1:
            raise ValueError("kernel_size must be a positive integer")
        self.kernel = np.ones(self.kernel_size) / self.kernel_size
        self.buffer = np.zeros(self.kernel_size)
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        if not isinstance(self.kernel_size, int) or self.kernel_size < 1:
            raise ValueError("kernel_size must be a positive integer")
        if not math.isfinite(self.v_threshold):
            raise ValueError("v_threshold must be finite")
        if self.kernel.shape != (self.kernel_size,) or not bool(np.isfinite(self.kernel).all()):
            raise ValueError("kernel must be finite with exactly kernel_size entries")
        if self.buffer.shape != (self.kernel_size,) or not bool(np.isfinite(self.buffer).all()):
            raise ValueError("buffer must be finite with exactly kernel_size entries")

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_configuration()

        self.buffer[self._ptr % self.kernel_size] = current
        self._ptr += 1
        n = min(self._ptr, self.kernel_size)
        score = float(np.dot(self.kernel[:n], self.buffer[:n]))
        if score >= self.v_threshold:
            self.buffer[:] = 0.0
            return 1
        return 0

    def reset(self) -> None:
        self.buffer[:] = 0.0
        self._ptr = 0
