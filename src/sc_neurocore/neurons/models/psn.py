# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Parallel Spiking Neuron (sliding PSN) — Fang et al. 2023

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ParallelSpikingNeuron:
    """k-order sliding Parallel Spiking Neuron — Fang et al. (2023).

    Streaming form of the PSN family (paper Eqs. 14–15):

    H[t] = sum_{i=0}^{k-1} W_i * X[t-k+1+i],  with X[j] = 0 for j < 0
    S[t] = Theta(H[t] - v_threshold)

    ``weights[i]`` is W_i, so ``weights[k-1]`` multiplies the newest
    input X[t] and ``weights[0]`` the oldest retained input X[t-k+1];
    the sum is accumulated sequentially from i = 0 to k-1 so every
    backend reproduces the same binary64 result bit-for-bit. The
    right-continuous ``Theta(0) = 1`` convention follows the paper. No
    PSN variant has a reset: firing never clears the input history, and
    :meth:`reset` only re-zeroes the retained inputs. The paper trains
    ``W`` and ``v_threshold`` per task and publishes no universal
    default; the uniform ``W_i = 1/k``, ``v_threshold = 1.0`` and
    ``kernel_size = 8`` defaults are repository defaults.

    Reference: Fang, W., Yu, Z., Zhou, Z., Chen, D., Chen, Y., Ma, Z.,
    Masquelier, T. & Tian, Y. (2023). Parallel Spiking Neurons
    with High Efficiency and Ability to Learn Long-term Dependencies.
    NeurIPS 2023. DOI 10.48550/arXiv.2304.12760.
    """

    kernel_size: int = 8
    v_threshold: float = 1.0
    weights: tuple[float, ...] = ()
    hidden: float = field(init=False, default=0.0)
    _history: list[float] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.kernel_size, int) or self.kernel_size < 1:
            raise ValueError("kernel_size must be a positive integer")
        if not self.weights:
            self.weights = tuple(1.0 / self.kernel_size for _ in range(self.kernel_size))
        self._history = [0.0] * self.kernel_size
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        if not isinstance(self.kernel_size, int) or self.kernel_size < 1:
            raise ValueError("kernel_size must be a positive integer")
        if len(self.weights) != self.kernel_size:
            raise ValueError("weights must have exactly kernel_size entries")
        if not all(math.isfinite(weight) for weight in self.weights):
            raise ValueError("weights must be finite")
        if not math.isfinite(self.v_threshold):
            raise ValueError("v_threshold must be finite")
        if len(self._history) != self.kernel_size or not all(
            math.isfinite(value) for value in self._history
        ):
            raise ValueError("retained inputs must be finite and of kernel_size length")

    def step(self, current: float) -> int:
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        self._validate_configuration()

        window = self._history[1:] + [current]
        hidden = 0.0
        for weight, value in zip(self.weights, window):
            hidden += weight * value
        if not math.isfinite(hidden):
            raise ValueError("sliding PSN hidden state became non-finite")

        self._history = window
        self.hidden = hidden
        return 1 if hidden >= self.v_threshold else 0

    def reset(self) -> None:
        self._history = [0.0] * self.kernel_size
        self.hidden = 0.0
