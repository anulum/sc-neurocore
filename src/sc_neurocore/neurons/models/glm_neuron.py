# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pillow et al. 2008 — generalized linear model

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class GLMNeuron:
    """Pillow et al. 2008 — generalized linear model (point-process GLM).

    lambda(t) = exp(k . stim(t) + h . spike_history(t) + mu)
    P(spike in dt) = lambda(t) * dt

    k: stimulus filter (length n_k)
    h: post-spike filter (length n_h), typically negative (refractoriness)

    Reference: Pillow, J.W. et al. (2008). Nature 454:1058–1062.
    """

    n_k: int = 10
    n_h: int = 20
    mu: float = -3.0
    dt_ms: float = 1.0
    k: NDArray[Any] = field(default=None, repr=False)  # type: ignore[arg-type]
    h: NDArray[Any] = field(default=None, repr=False)  # type: ignore[arg-type]
    _stim_buf: NDArray[Any] = field(default=None, repr=False)  # type: ignore[arg-type]
    _spike_buf: NDArray[Any] = field(default=None, repr=False)  # type: ignore[arg-type]
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        if self.k is None:
            self.k = np.exp(-np.arange(self.n_k) / 3.0) * 0.5
        if self.h is None:
            t = np.arange(self.n_h)
            self.h = -5.0 * np.exp(-t / 2.0) + 0.5 * np.exp(-t / 10.0)
        self._stim_buf = np.zeros(self.n_k)
        self._spike_buf = np.zeros(self.n_h)
        self._rng = np.random.default_rng()

    def step(self, stimulus: float) -> int:
        self._stim_buf = np.roll(self._stim_buf, 1)
        self._stim_buf[0] = stimulus
        log_rate = float(np.dot(self.k, self._stim_buf) + np.dot(self.h, self._spike_buf) + self.mu)
        lam = np.exp(np.clip(log_rate, -20.0, 20.0))
        p = lam * self.dt_ms / 1000.0
        spike = 1 if self._rng.random() < min(p, 1.0) else 0
        self._spike_buf = np.roll(self._spike_buf, 1)
        self._spike_buf[0] = float(spike)
        return spike

    def reset(self) -> None:
        self._stim_buf = np.zeros(self.n_k)
        self._spike_buf = np.zeros(self.n_h)
