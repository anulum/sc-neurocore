# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — source-faithful Nagumo–Sato refractory map

"""Source-faithful Nagumo–Sato one-state neuron map.

Aihara's primary-author review reproduces the Nagumo–Sato reduction as

``y[t+1] = k*y[t] - alpha*H(y[t]) + bias + current[t]``
``x[t+1] = H(y[t+1])``

where ``H(z)=1`` for ``z >= 0`` and zero otherwise. ``current`` is an
additive perturbation of the transformed stimulus ``a(t)``; it is not the raw
historical stimulus ``A(t)`` from the infinite-memory form.

References
----------
Nagumo, J. & Sato, S. (1972). *On a response characteristic of a
mathematical neuron model*. Kybernetik 10, 155–164.
https://doi.org/10.1007/BF00290514
Aihara, K. (1989). *Chaotic Neural Networks*. RIMS Kokyuroku 710,
145–163, Eqs. 1–7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

NAGUMO_SATO_INITIAL_Y = 0.1
NagumoSatoMapResult = dict[str, npt.NDArray[np.float64] | float | int]


@dataclass
class NagumoSatoMapNeuron:
    """Nagumo and Sato's discontinuous refractory neuron map.

    Parameters
    ----------
    y : float, default=0.1
        Current internal state, matching Aihara's Figure 3 initial condition.
    k : float, default=0.6
        Refractory-memory damping factor; the source requires ``0 <= k < 1``.
    alpha : float, default=1.0
        Positive refractory decrement following a firing output.
    bias : float, default=0.2
        Constant transformed stimulus ``a``.
    """

    y: float = NAGUMO_SATO_INITIAL_Y
    k: float = 0.6
    alpha: float = 1.0
    bias: float = 0.2

    def __post_init__(self) -> None:
        """Normalise scalar fields and reject invalid configuration."""
        for name in ("y", "k", "alpha", "bias"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            setattr(self, name, value)
        self._validated_state()
        self._validated_parameters()

    @staticmethod
    def _heaviside(value: float) -> int:
        return int(value >= 0.0)

    def _validated_state(self) -> float:
        try:
            value = float(self.y)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Nagumo-Sato internal state must be numeric") from exc
        if not math.isfinite(value):
            raise FloatingPointError("Nagumo-Sato internal state must be finite")
        return value

    def _validated_parameters(self) -> tuple[float, float, float]:
        try:
            values = (float(self.k), float(self.alpha), float(self.bias))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Nagumo-Sato parameters must be numeric") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Nagumo-Sato parameters must be finite")
        k, alpha, bias = values
        if not 0.0 <= k < 1.0:
            raise ValueError("k must satisfy 0 <= k < 1")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        return k, alpha, bias

    @property
    def x(self) -> int:
        """Return the current all-or-none firing output ``H(y)``."""
        return self.output()

    def output(self) -> int:
        """Return the current all-or-none firing output ``H(y)``."""
        return self._heaviside(self._validated_state())

    def _candidate(self, current: float) -> tuple[float, int]:
        try:
            drive = float(current)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("current must be numeric") from exc
        if not math.isfinite(drive):
            raise ValueError("current must be finite")
        y = self._validated_state()
        k, alpha, bias = self._validated_parameters()
        next_y = k * y - alpha * self._heaviside(y) + bias + drive
        if not math.isfinite(next_y):
            raise FloatingPointError("Nagumo-Sato map candidate must be finite")
        return next_y, self._heaviside(next_y)

    def step(self, current: float = 0.0) -> int:
        """Advance one source-equation step and return ``H(y[t+1])``."""
        next_y, event = self._candidate(current)
        self.y = next_y
        return event

    def simulate(
        self,
        current: npt.ArrayLike,
        *,
        backend: str = "auto",
    ) -> NagumoSatoMapResult:
        """Run an atomic batch on Python, Rust, Julia, Go, or Mojo."""
        from sc_neurocore.accel.nagumo_sato_map import simulate_nagumo_sato_map

        result = simulate_nagumo_sato_map(
            self.y,
            self.k,
            self.alpha,
            self.bias,
            current,
            backend=backend,
        )
        self.y = float(cast(float, result["y_final"]))
        return result

    def reset(self) -> None:
        """Restore the source initial state while preserving parameters."""
        self.y = NAGUMO_SATO_INITIAL_Y


__all__ = ["NAGUMO_SATO_INITIAL_Y", "NagumoSatoMapNeuron", "NagumoSatoMapResult"]
