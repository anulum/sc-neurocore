# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Izhikevich resonate-and-fire neuron

"""Source-bound resonate-and-fire dynamics with exact linear subthreshold flow."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

ResonateAndFireResult = dict[str, npt.NDArray[np.float64] | float | int]


@dataclass
class ResonateAndFireNeuron:
    """Damped complex resonator from Izhikevich (2001).

    The source defines ``z = x + i y`` and

    ``dz/dt = (b + i * omega) * z + I``.

    ``x`` is current-like and ``y`` is voltage-like. A spike is emitted on an
    upward sampled crossing of ``y = threshold``. The source post-spike reset
    is ``z = i``; the generalized implementation therefore installs
    ``(x, y) = (0, threshold)``. The maintained numerical step is the exact
    constant-real-input flow over one ``dt`` interval.

    Defaults ``b=-1``, ``omega=10``, and ``threshold=1`` are the parameter set
    used for most illustrations in the source paper. ``dt=0.01`` is an
    implementation sampling interval, not a parameter asserted by the paper.

    Reference
    ---------
    Izhikevich, E. M. (2001). Resonate-and-fire neurons.
    Neural Networks 14(6-7), 883-894.
    https://doi.org/10.1016/S0893-6080(01)00078-8
    """

    x: float = 0.0
    y: float = 0.0
    b: float = -1.0
    omega: float = 10.0
    threshold: float = 1.0
    dt: float = 0.01

    def __post_init__(self) -> None:
        """Normalise scalar fields and reject an invalid configuration."""
        for name in ("x", "y", "b", "omega", "threshold", "dt"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            setattr(self, name, value)
        self._validated_state()
        self._validated_parameters()

    def _validated_state(self) -> tuple[float, float]:
        """Return the current finite current-like and voltage-like states."""
        try:
            x = float(self.x)
            y = float(self.y)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("resonate-and-fire state must be numeric") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("resonate-and-fire state must be finite")
        return x, y

    def _validated_parameters(self) -> tuple[float, float, float, float]:
        """Return the finite numerical configuration without mutation."""
        try:
            b = float(self.b)
            omega = float(self.omega)
            threshold = float(self.threshold)
            dt = float(self.dt)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("resonate-and-fire parameters must be numeric") from exc
        if not all(math.isfinite(value) for value in (b, omega, threshold, dt)):
            raise ValueError("resonate-and-fire parameters must be finite")
        if omega <= 0.0:
            raise ValueError("omega must be positive")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        return b, omega, threshold, dt

    @staticmethod
    def _exact_linear_flow(
        x: float,
        y: float,
        current: float,
        b: float,
        omega: float,
        dt: float,
    ) -> tuple[float, float]:
        """Return the exact piecewise-constant-input linear-flow candidate."""
        denominator = b * b + omega * omega
        damping_argument = b * dt
        angle = omega * dt
        if not all(math.isfinite(value) for value in (denominator, damping_argument, angle)):
            raise FloatingPointError("exact resonator coefficients must be finite")
        if denominator <= 0.0:
            raise ValueError("exact resonator denominator must be positive")

        x_ss = -b * current / denominator
        y_ss = omega * current / denominator
        if not math.isfinite(x_ss) or not math.isfinite(y_ss):
            raise FloatingPointError("exact resonator equilibrium must be finite")
        try:
            decay = math.exp(damping_argument)
        except OverflowError as exc:
            raise FloatingPointError("exact resonator decay must be finite") from exc
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        dx = x - x_ss
        dy = y - y_ss
        next_x = x_ss + decay * (dx * cos_angle - dy * sin_angle)
        next_y = y_ss + decay * (dx * sin_angle + dy * cos_angle)
        if not math.isfinite(next_x) or not math.isfinite(next_y):
            raise FloatingPointError("exact resonator candidate must be finite")
        return next_x, next_y

    def _candidate(self, current: float) -> tuple[float, float, int]:
        """Compute one validated candidate without mutating caller-visible state."""
        try:
            drive = float(current)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("current must be numeric") from exc
        if not math.isfinite(drive):
            raise ValueError("current must be finite")
        x, y = self._validated_state()
        b, omega, threshold, dt = self._validated_parameters()
        next_x, next_y = self._exact_linear_flow(x, y, drive, b, omega, dt)
        if y < threshold <= next_y:
            return 0.0, threshold, 1
        return next_x, next_y, 0

    def step(self, current: float = 0.0) -> int:
        """Advance one exact-flow interval and return a binary spike event.

        Mutation is atomic: invalid input, configuration, or candidate state
        leaves both dynamic states unchanged.
        """
        next_x, next_y, spike = self._candidate(current)
        self.x, self.y = next_x, next_y
        return spike

    def simulate(
        self,
        current: npt.ArrayLike,
        *,
        backend: str = "auto",
    ) -> ResonateAndFireResult:
        """Run one atomic piecewise-constant-input batch on a maintained backend."""
        from sc_neurocore.accel.resonate_and_fire import simulate_resonate_and_fire

        result = simulate_resonate_and_fire(
            self.x,
            self.y,
            self.b,
            self.omega,
            self.threshold,
            self.dt,
            current,
            backend=backend,
        )
        self.x = float(cast(float, result["x_final"]))
        self.y = float(cast(float, result["y_final"]))
        return result

    def reset(self) -> None:
        """Restore the quiescent initial state while preserving parameters."""
        self.x = 0.0
        self.y = 0.0


__all__ = ["ResonateAndFireNeuron", "ResonateAndFireResult"]
