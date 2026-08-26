# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pillow et al. 2008 — generalized linear model

from __future__ import annotations

import math
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

    Reference: Pillow, J.W. et al. (2008). Nature 454:995–999.
    The exponential nonlinearity, log-rate clip to [-20, 20], and
    Bernoulli per-bin sampling are the repository's discrete-time
    specialisation of the paper's point process. Pass ``seed`` for a
    reproducible generator, and ``uniform`` to :meth:`step` to supply
    the Bernoulli sample explicitly (exact cross-backend parity).
    """

    n_k: int = 10
    n_h: int = 20
    mu: float = -3.0
    dt_ms: float = 1.0
    seed: int | None = None
    k: NDArray[Any] = field(default=None, repr=False)  # type: ignore[arg-type]
    h: NDArray[Any] = field(default=None, repr=False)  # type: ignore[arg-type]
    _stim_buf: NDArray[Any] = field(default=None, repr=False)  # type: ignore[arg-type]
    _spike_buf: NDArray[Any] = field(default=None, repr=False)  # type: ignore[arg-type]
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.n_k, int) or isinstance(self.n_k, bool):
            raise TypeError("n_k must be an integer")
        if not isinstance(self.n_h, int) or isinstance(self.n_h, bool):
            raise TypeError("n_h must be an integer")
        if not (1 <= self.n_k <= 10_000 and 1 <= self.n_h <= 10_000):
            raise ValueError("n_k and n_h must be within [1, 10000]")
        if self.k is None:
            self.k = np.exp(-np.arange(self.n_k) / 3.0) * 0.5
        if self.h is None:
            t = np.arange(self.n_h)
            self.h = -5.0 * np.exp(-t / 2.0) + 0.5 * np.exp(-t / 10.0)
        self.k = np.asarray(self.k, dtype=np.float64)
        self.h = np.asarray(self.h, dtype=np.float64)
        self._stim_buf = np.zeros(self.n_k)
        self._spike_buf = np.zeros(self.n_h)
        self._rng = np.random.default_rng(self.seed)
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        if not math.isfinite(self.mu):
            raise ValueError("mu must be finite")
        if not (math.isfinite(self.dt_ms) and 0.0 < self.dt_ms <= 1000.0):
            raise ValueError("dt_ms must be finite and within (0, 1000] ms")
        if self.k.shape != (self.n_k,) or self.h.shape != (self.n_h,):
            raise ValueError("k and h must have lengths n_k and n_h")
        if not (np.all(np.isfinite(self.k)) and np.all(np.isfinite(self.h))):
            raise ValueError("k and h must be finite")
        if self._stim_buf.shape != (self.n_k,) or self._spike_buf.shape != (self.n_h,):
            raise ValueError("history buffers must have lengths n_k and n_h")
        if not (np.all(np.isfinite(self._stim_buf)) and np.all(np.isfinite(self._spike_buf))):
            raise ValueError("history buffers must be finite")

    def step(self, stimulus: float, uniform: float | None = None) -> int:
        if not math.isfinite(stimulus):
            raise ValueError("stimulus must be finite")
        if uniform is not None and not (math.isfinite(uniform) and 0.0 <= uniform < 1.0):
            raise ValueError("uniform must be finite and within [0, 1)")
        self._validate_configuration()

        stim_candidate = np.roll(self._stim_buf, 1)
        stim_candidate[0] = stimulus
        log_rate = float(np.dot(self.k, stim_candidate) + np.dot(self.h, self._spike_buf) + self.mu)
        lam = np.exp(np.clip(log_rate, -20.0, 20.0))
        p = lam * self.dt_ms / 1000.0
        draw = self._rng.random() if uniform is None else uniform
        spike = 1 if draw < min(p, 1.0) else 0
        spike_candidate = np.roll(self._spike_buf, 1)
        spike_candidate[0] = float(spike)

        self._stim_buf = stim_candidate
        self._spike_buf = spike_candidate
        return spike

    def reset(self) -> None:
        self._stim_buf = np.zeros(self.n_k)
        self._spike_buf = np.zeros(self.n_h)
