# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Learning rate schedulers for SNN training

"""Learning rate schedulers for SNN training.

Usage::

    from sc_neurocore.learning.schedulers import CosineScheduler

    sched = CosineScheduler(lr_init=0.01, lr_min=1e-4, total_steps=1000)
    for step in range(1000):
        lr = sched.step()
        synapse.learning_rate = lr
"""

from __future__ import annotations

import math


class StepScheduler:
    """Drop learning rate by *gamma* every *step_size* steps."""

    def __init__(self, lr_init: float, step_size: int, gamma: float = 0.1):
        self.lr = lr_init
        self.step_size = step_size
        self.gamma = gamma
        self._count = 0

    def step(self) -> float:
        """Advance one scheduler step and return the current learning rate."""
        self._count += 1
        if self._count % self.step_size == 0:
            self.lr *= self.gamma
        return self.lr

    def reset(self) -> None:
        """Reset the internal step counter without changing the current rate."""
        self._count = 0


class ExponentialScheduler:
    """Multiply learning rate by *gamma* each step."""

    def __init__(self, lr_init: float, gamma: float = 0.999):
        self.lr = lr_init
        self.gamma = gamma

    def step(self) -> float:
        """Apply one exponential decay update and return the new rate."""
        self.lr *= self.gamma
        return self.lr

    def reset(self) -> None:
        """Leave the stateless exponential schedule unchanged."""
        pass


class CosineScheduler:
    """Cosine annealing from *lr_init* to *lr_min* over *total_steps*."""

    def __init__(self, lr_init: float, lr_min: float, total_steps: int):
        self.lr_init = lr_init
        self.lr_min = lr_min
        self.total_steps = total_steps
        self._count = 0
        self.lr = lr_init

    def step(self) -> float:
        """Advance one cosine-annealing step and return the new rate."""
        self._count += 1
        t = min(self._count / self.total_steps, 1.0)
        self.lr = self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (1 + math.cos(math.pi * t))
        return self.lr

    def reset(self) -> None:
        """Restore the initial learning rate and restart the cosine schedule."""
        self._count = 0
        self.lr = self.lr_init


class WarmupCosineScheduler:
    """Linear warmup followed by cosine decay."""

    def __init__(
        self,
        lr_init: float,
        lr_min: float,
        warmup_steps: int,
        total_steps: int,
    ):
        self.lr_init = lr_init
        self.lr_min = lr_min
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self._count = 0
        self.lr = 0.0

    def step(self) -> float:
        """Advance through warmup or cosine decay and return the current rate."""
        self._count += 1
        if self._count <= self.warmup_steps:
            self.lr = self.lr_init * (self._count / self.warmup_steps)
        else:
            decay_steps = self.total_steps - self.warmup_steps
            t = min((self._count - self.warmup_steps) / decay_steps, 1.0)
            self.lr = self.lr_min + 0.5 * (self.lr_init - self.lr_min) * (1 + math.cos(math.pi * t))
        return self.lr

    def reset(self) -> None:
        """Return to the pre-warmup state with zero current learning rate."""
        self._count = 0
        self.lr = 0.0
