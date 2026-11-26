# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — KLIF — LIF with learnable scaling factor k

"""KLIF leaky integrate-and-fire neuron with learnable input gain ``k``."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real

_isfinite = math.isfinite


def _require_finite_float(value: object, name: str) -> float:
    """Validate that ``value`` is a finite real number and return it as ``float``."""
    if type(value) is float:
        result = value
    elif isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
    else:
        result = float(value)
    if not _isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return result


def _require_positive_float(value: object, name: str) -> float:
    """Validate that ``value`` is a strictly positive finite real number."""
    result = _require_finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return result


@dataclass
class KLIFNeuron:
    """KLIF — LIF with learnable scaling factor k.

    V[t+1] = alpha * V[t] + k * I; spike when V >= threshold.
    The scaling factor k is a trainable parameter for SNN backprop.

    Reference: Jiang, C. & Zhang, Y. (2024). Neural Comput. 36(8):1546–1564.

    Parameters
    ----------
    v : float, default=0.0
        Initial membrane potential.
    k : float, default=1.0
        Trainable input scaling factor.
    tau : float, default=10.0
        Membrane time constant in milliseconds; must be strictly positive.
    v_threshold : float, default=1.0
        Firing threshold.
    v_reset : float, default=0.0
        Post-spike reset potential.
    dt : float, default=1.0
        Discrete integration timestep in milliseconds; must be strictly positive.

    Raises
    ------
    TypeError
        If any parameter is a boolean or not a real number.
    ValueError
        If any parameter is non-finite, or if ``tau <= 0`` or ``dt <= 0``.
    """

    v: float = 0.0
    k: float = 1.0  # learnable scaling factor
    tau: float = 10.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    dt: float = 1.0
    alpha: float = field(init=False)

    def __post_init__(self) -> None:
        """Validate initial fields and precompute exponential decay ``alpha``."""
        self.v = _require_finite_float(self.v, "v")
        self.k = _require_finite_float(self.k, "k")
        self.tau = _require_positive_float(self.tau, "tau")
        self.v_threshold = _require_finite_float(self.v_threshold, "v_threshold")
        self.v_reset = _require_finite_float(self.v_reset, "v_reset")
        self.dt = _require_positive_float(self.dt, "dt")
        self.alpha = math.exp(-self.dt / self.tau)

    def step(self, current: float) -> int:
        """Advance the neuron by one discrete timestep.

        Parameters
        ----------
        current : float
            External input drive scaled by ``k``.

        Returns
        -------
        int
            ``1`` if the updated membrane potential reaches ``v_threshold``, else ``0``.

        Raises
        ------
        TypeError
            If ``current`` or any neuron field is a boolean or not a real number.
        ValueError
            If ``current`` or any field is non-finite, if ``tau <= 0`` or ``dt <= 0``,
            or if the candidate membrane potential overflows; state is unchanged on error.
        """
        v = self.v
        alpha = self.alpha
        k = self.k
        tau = self.tau
        v_threshold = self.v_threshold
        v_reset = self.v_reset
        dt = self.dt
        if not (
            current.__class__ is float
            and v.__class__ is float
            and alpha.__class__ is float
            and k.__class__ is float
            and tau.__class__ is float
            and v_threshold.__class__ is float
            and v_reset.__class__ is float
            and dt.__class__ is float
            and (current + v + alpha + k + tau + v_threshold + v_reset + dt) * 0.0 == 0.0
            and tau > 0.0
            and dt > 0.0
        ):
            current = _require_finite_float(current, "current")
            v = _require_finite_float(v, "v")
            alpha = _require_finite_float(alpha, "alpha")
            k = _require_finite_float(k, "k")
            _require_positive_float(tau, "tau")
            v_threshold = _require_finite_float(v_threshold, "v_threshold")
            v_reset = _require_finite_float(v_reset, "v_reset")
            _require_positive_float(dt, "dt")
        next_v = alpha * v + k * current
        if not _isfinite(next_v):
            raise ValueError(f"updated membrane potential v must be finite, got {next_v!r}.")
        if next_v >= v_threshold:
            self.v = v_reset
            return 1
        self.v = next_v
        return 0

    def reset(self) -> None:
        """Reset membrane potential to rest."""
        self.v = 0.0
