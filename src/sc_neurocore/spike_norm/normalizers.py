# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SNN normalization layer implementations

"""5 SNN normalization variants. No framework ships these as reusable modules.

tdBN: threshold-dependent BN (Zheng 2021)
BNTT: per-timestep BN (Kim & Panda 2021)
TEBN: temporal effective BN (Duan 2022, NeurIPS)
MPBN: membrane potential BN with inference re-parameterization (Guo 2023, ICCV)
TAB: temporal accumulated BN (Jiang 2024, ICLR)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ThresholdDependentBN:
    """tdBN: incorporates firing threshold into normalization.

    BN(x) = gamma * (x - mean) / sqrt(var + eps) + beta
    where mean/var are computed across batch, adjusted by V_threshold.

    Parameters
    ----------
    n_features : int
    threshold : float
    momentum : float
    """

    n_features: int
    threshold: float = 1.0
    momentum: float = 0.1
    eps: float = 1e-5

    def __post_init__(self) -> None:
        self.gamma = np.ones(self.n_features)
        self.beta = np.zeros(self.n_features)
        self.running_mean = np.zeros(self.n_features)
        self.running_var = np.ones(self.n_features)

    def forward(self, x: np.ndarray[Any, Any], training: bool = True) -> np.ndarray[Any, Any]:
        if training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        result: np.ndarray[Any, Any] = self.gamma * x_norm * self.threshold + self.beta
        return result


@dataclass
class PerTimestepBN:
    """BNTT: separate BN statistics per timestep.

    Each timestep t has its own mean_t, var_t, gamma_t, beta_t.

    Parameters
    ----------
    n_features : int
    T : int
        Number of timesteps.
    """

    n_features: int
    T: int
    eps: float = 1e-5

    def __post_init__(self) -> None:
        self.gammas = [np.ones(self.n_features) for _ in range(self.T)]
        self.betas = [np.zeros(self.n_features) for _ in range(self.T)]
        self.running_means = [np.zeros(self.n_features) for _ in range(self.T)]
        self.running_vars = [np.ones(self.n_features) for _ in range(self.T)]

    def forward(self, x: np.ndarray[Any, Any], t: int, training: bool = True) -> np.ndarray[Any, Any]:
        t_idx = min(t, self.T - 1)
        if training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            self.running_means[t_idx] = 0.9 * self.running_means[t_idx] + 0.1 * mean
            self.running_vars[t_idx] = 0.9 * self.running_vars[t_idx] + 0.1 * var
        else:  # pragma: no cover
            mean = self.running_means[t_idx]
            var = self.running_vars[t_idx]
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        result: np.ndarray[Any, Any] = self.gammas[t_idx] * x_norm + self.betas[t_idx]
        return result


@dataclass
class TemporalEffectiveBN:
    """TEBN: rescales presynaptic inputs per timestep.

    Applies BN then per-timestep scaling factor lambda_t.

    Parameters
    ----------
    n_features : int
    T : int
    """

    n_features: int
    T: int
    eps: float = 1e-5

    def __post_init__(self) -> None:
        self.gamma = np.ones(self.n_features)
        self.beta = np.zeros(self.n_features)
        self.lambdas = np.ones(self.T)
        self.running_mean = np.zeros(self.n_features)
        self.running_var = np.ones(self.n_features)

    def forward(self, x: np.ndarray[Any, Any], t: int, training: bool = True) -> np.ndarray[Any, Any]:
        if training:
            mean = x.mean(axis=0)
            var = x.var(axis=0)
            self.running_mean = 0.9 * self.running_mean + 0.1 * mean
            self.running_var = 0.9 * self.running_var + 0.1 * var
        else:  # pragma: no cover
            mean = self.running_mean
            var = self.running_var
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        t_idx = min(t, self.T - 1)
        result: np.ndarray[Any, Any] = self.lambdas[t_idx] * (self.gamma * x_norm + self.beta)
        return result


@dataclass
class MembranePotentialBN:
    """MPBN: BN on membrane potential before spike function.

    At inference: fold BN into threshold (zero overhead).
    new_threshold = (V_th - beta) * sqrt(var + eps) / gamma + mean

    Parameters
    ----------
    n_features : int
    threshold : float
    """

    n_features: int
    threshold: float = 1.0
    momentum: float = 0.1
    eps: float = 1e-5

    def __post_init__(self) -> None:
        self.gamma = np.ones(self.n_features)
        self.beta = np.zeros(self.n_features)
        self.running_mean = np.zeros(self.n_features)
        self.running_var = np.ones(self.n_features)

    def forward(self, membrane: np.ndarray[Any, Any], training: bool = True) -> np.ndarray[Any, Any]:
        if training:
            mean = membrane.mean(axis=0) if membrane.ndim > 1 else membrane
            var = membrane.var(axis=0) if membrane.ndim > 1 else np.zeros_like(membrane)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            norm = (membrane - mean) / np.sqrt(var + self.eps)
            result: np.ndarray[Any, Any] = self.gamma * norm + self.beta
            return result
        return membrane

    def fused_threshold(self) -> np.ndarray[Any, Any]:
        """Compute per-neuron threshold that absorbs BN at inference.

        Returns ndarray of shape (n_features,) — use as per-neuron threshold
        instead of applying BN at inference (zero overhead).
        """
        result: np.ndarray[Any, Any] = (
            (self.threshold - self.beta) * np.sqrt(self.running_var + self.eps) / np.clip(
                self.gamma, 1e-8, None
            ) + self.running_mean
        )
        return result


@dataclass
class TemporalAccumulatedBN:
    """TAB: normalizes accumulated membrane potential.

    Tracks running accumulated potential across timesteps.
    Addresses Temporal Covariate Shift directly.

    Parameters
    ----------
    n_features : int
    """

    n_features: int
    momentum: float = 0.1
    eps: float = 1e-5

    def __post_init__(self) -> None:
        self.gamma = np.ones(self.n_features)
        self.beta = np.zeros(self.n_features)
        self.running_mean = np.zeros(self.n_features)
        self.running_var = np.ones(self.n_features)
        self._accumulated = np.zeros(self.n_features)

    def forward(self, x: np.ndarray[Any, Any], training: bool = True) -> np.ndarray[Any, Any]:
        increment: np.ndarray[Any, Any] = x.mean(axis=0) if x.ndim > 1 else x
        self._accumulated = self._accumulated + increment
        if training:
            mean = self._accumulated
            # Variance estimated from current input
            var = x.var(axis=0) if x.ndim > 1 else np.zeros_like(x)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:  # pragma: no cover
            mean = self.running_mean
            var = self.running_var
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        result: np.ndarray[Any, Any] = self.gamma * x_norm + self.beta
        return result

    def reset(self) -> None:
        self._accumulated = np.zeros(self.n_features)
