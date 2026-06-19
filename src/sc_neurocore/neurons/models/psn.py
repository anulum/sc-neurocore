# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Parallel Spiking Neuron — 2024, linear filter over all

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ParallelSpikingNeuron:
    """Parallel Spiking Neuron — 2024, linear filter over all timesteps.

    Applies a learned 1D convolution kernel over an internal buffer,
    enabling non-causal temporal aggregation during training.
    At each step: score = sum(kernel * buffer); spike if score >= threshold.

    Reference: Comsa, I.-M. et al. (2020). Proc. ICLR 2020.
    """

    kernel_size: int = 8
    v_threshold: float = 1.0
    kernel: np.ndarray[Any, Any] = field(init=False)
    buffer: np.ndarray[Any, Any] = field(init=False)
    _ptr: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.kernel = np.ones(self.kernel_size) / self.kernel_size
        self.buffer = np.zeros(self.kernel_size)

    def step(self, current: float) -> int:
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
