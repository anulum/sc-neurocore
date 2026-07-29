# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Aihara–Takabe–Toyoda chaotic neuron map

"""Source-faithful Aihara chaotic neuron with a graded output.

The primary-author recurrence is the one-state discrete map

``y[t+1] = k*y[t] - alpha*f(y[t]) + bias + current[t]``

with ``f(y) = 1 / (1 + exp(-y/epsilon))``. The independent state is ``y``;
``x=f(y)`` is its graded output. The binary event follows Aihara's waveform
shaper, ``h(x)=1`` for ``x >= 0.5`` and zero otherwise.

``current`` is the effective additive stimulus in the reduced map, not the raw
historical input ``A(t)`` whose temporal reduction also contains ``A(t-1)``.

References
----------
Aihara, K. (1989). *Chaotic Neural Networks*. RIMS Kokyuroku 710, 145–163,
Eqs. 8–12. The manuscript cites the Aihara–Takabe–Toyoda article as submitted.
Aihara, K., Takabe, T. & Toyoda, M. (1990). *Chaotic neural networks*.
Physics Letters A 144(6–7), 333–340.
https://doi.org/10.1016/0375-9601(90)90136-C
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

AIHARA_INITIAL_Y = 0.1
AIHARA_CHAOTIC_BIAS = 0.3968
AiharaMapResult = dict[str, npt.NDArray[np.float64] | float | int]


@dataclass
class AiharaMapNeuron:
    """Aihara's graded one-state chaotic neuron.

    Parameters
    ----------
    y : float, default=0.1
        Current internal state. The default is the initial condition used for
        the source parameter sweep.
    k : float, default=0.7
        Refractory-memory decay factor; the source requires ``0 <= k < 1``.
    alpha : float, default=1.0
        Positive refractory scaling coefficient.
    bias : float, default=0.3968
        Constant effective stimulus ``a``. The default is the chaotic example
        in the primary-author manuscript's Figure 4.
    epsilon : float, default=0.01
        Positive logistic steepness. The default is the Figure 4 value.
    """

    y: float = AIHARA_INITIAL_Y
    k: float = 0.7
    alpha: float = 1.0
    bias: float = AIHARA_CHAOTIC_BIAS
    epsilon: float = 0.01

    def __post_init__(self) -> None:
        """Normalise scalar fields and reject an invalid configuration."""
        for name in ("y", "k", "alpha", "bias", "epsilon"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            setattr(self, name, value)
        self._validated_state()
        self._validated_parameters()

    def _validated_state(self) -> float:
        try:
            value = float(self.y)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Aihara internal state must be numeric") from exc
        if not math.isfinite(value):
            raise FloatingPointError("Aihara internal state must be finite")
        return value

    def _validated_parameters(self) -> tuple[float, float, float, float]:
        try:
            values = (
                float(self.k),
                float(self.alpha),
                float(self.bias),
                float(self.epsilon),
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Aihara parameters must be numeric") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Aihara parameters must be finite")
        k, alpha, bias, epsilon = values
        if not 0.0 <= k < 1.0:
            raise ValueError("k must satisfy 0 <= k < 1")
        if alpha <= 0.0:
            raise ValueError("alpha must be positive")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        return k, alpha, bias, epsilon

    @staticmethod
    def _logistic(value: float, epsilon: float) -> float:
        argument = value / epsilon
        if argument >= 0.0:
            return 1.0 / (1.0 + math.exp(-argument))
        exponential = math.exp(argument)
        return exponential / (1.0 + exponential)

    @property
    def x(self) -> float:
        """Return the source graded output ``x=f(y)``."""
        return self.output()

    def output(self) -> float:
        """Return the source graded output ``x=f(y)``."""
        return self._logistic(self._validated_state(), self.epsilon)

    def _candidate(self, current: float) -> tuple[float, float, int]:
        try:
            drive = float(current)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("current must be numeric") from exc
        if not math.isfinite(drive):
            raise ValueError("current must be finite")
        y = self._validated_state()
        k, alpha, bias, epsilon = self._validated_parameters()
        output = self._logistic(y, epsilon)
        next_y = k * y - alpha * output + bias + drive
        if not math.isfinite(next_y):
            raise FloatingPointError("Aihara map candidate must be finite")
        next_output = self._logistic(next_y, epsilon)
        return next_y, next_output, int(next_output >= 0.5)

    def step(self, current: float = 0.0) -> int:
        """Advance one map step and return the source thresholded output.

        Parameters
        ----------
        current : float, default=0.0
            Effective additive stimulus for this discrete step.

        Returns
        -------
        int
            ``1`` when the new graded output is at least ``0.5``; otherwise
            ``0``.

        Raises
        ------
        ValueError
            If the input or mutable configuration is invalid.
        FloatingPointError
            If the state or candidate is non-finite.
        """
        next_y, _next_output, event = self._candidate(current)
        self.y = next_y
        return event

    def simulate(
        self,
        current: npt.ArrayLike,
        *,
        backend: str = "auto",
    ) -> AiharaMapResult:
        """Run an atomic piecewise-stimulus batch on a maintained backend.

        Parameters
        ----------
        current : ArrayLike
            One finite effective stimulus per discrete step.
        backend : {"auto", "python", "rust", "julia", "go", "mojo"}
            Execution lane. ``auto`` uses committed measured ordering.

        Returns
        -------
        dict
            Complete ``y``, graded ``x``, and binary ``spikes`` trajectories,
            plus final-state and event-count receipts.

        Raises
        ------
        ValueError
            If the configuration, input vector, or backend is invalid.
        RuntimeError
            If an explicitly requested backend is unavailable.
        FloatingPointError
            If a candidate or backend receipt violates the model contract.
        """
        from sc_neurocore.accel.aihara_map import simulate_aihara_map

        result = simulate_aihara_map(
            self.y,
            self.k,
            self.alpha,
            self.bias,
            self.epsilon,
            current,
            backend=backend,
        )
        self.y = float(cast(float, result["y_final"]))
        return result

    def reset(self) -> None:
        """Restore the source initial state while preserving parameters."""
        self.y = AIHARA_INITIAL_Y


__all__ = [
    "AIHARA_CHAOTIC_BIAS",
    "AIHARA_INITIAL_Y",
    "AiharaMapNeuron",
    "AiharaMapResult",
]
