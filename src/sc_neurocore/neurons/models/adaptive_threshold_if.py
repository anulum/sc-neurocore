# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Composite reduced adaptive-threshold leaky integrate-and-fire neuron

"""Exact-relaxation leaky integrate-and-fire with a decaying adaptive threshold."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

AdaptiveThresholdIFResult = dict[str, npt.NDArray[np.float64] | float | int]


@dataclass
class AdaptiveThresholdIFNeuron:
    """Composite reduced adaptive-threshold leaky integrate-and-fire neuron.

    The membrane equation is the leaky integrate-and-fire relaxation

    ``tau_m * dV/dt = -(V - v_rest) + I``,

    integrated with the exact constant-input flow over one ``dt`` interval.
    The threshold equation is

    ``dtheta/dt = -(theta - theta_rest) / tau_theta``,

    which is the Mihalas and Niebur (2009) threshold equation
    ``dTheta/dt = a(V - E_L) - b(Theta - Theta_inf)`` taken at zero voltage
    coupling (``a = 0``), integrated with the same exact relaxation. A spike
    is emitted when the candidate membrane potential reaches the candidate
    threshold; the membrane potential then resets to ``v_reset`` and the
    threshold increases by the fixed amount ``delta_theta`` — the fixed
    post-spike threshold shift derived in Platkiewicz and Brette (2010).

    Reduction boundary: the Mihalas–Niebur voltage-coupling term ``a(V - E_L)``
    and the Platkiewicz–Brette voltage-dependent threshold equilibrium
    ``theta_inf(V)`` are outside this reduced model, as is any adaptation
    current. Defaults are catalogue/model-family choices, not source-derived
    parameters.

    References
    ----------
    Mihalas, S. and Niebur, E. (2009). A generalized linear integrate-and-fire
    neural model produces diverse spiking behaviors. Neural Computation 21(3),
    704–718. https://doi.org/10.1162/neco.2008.12-07-680

    Platkiewicz, J. and Brette, R. (2010). A threshold equation for action
    potential initiation. PLoS Computational Biology 6(7), e1000850.
    https://doi.org/10.1371/journal.pcbi.1000850
    """

    v: float = -65.0
    theta: float = -50.0
    v_rest: float = -65.0
    v_reset: float = -65.0
    theta_rest: float = -50.0
    delta_theta: float = 5.0
    tau_m: float = 10.0
    tau_theta: float = 50.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        """Normalise scalar fields and reject an invalid configuration."""
        for name in (
            "v",
            "theta",
            "v_rest",
            "v_reset",
            "theta_rest",
            "delta_theta",
            "tau_m",
            "tau_theta",
            "dt",
        ):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must be numeric") from exc
            setattr(self, name, value)
        self._validated_state()
        self._validated_parameters()

    def _validated_state(self) -> tuple[float, float]:
        """Return the current finite voltage and threshold states."""
        try:
            v = float(self.v)
            theta = float(self.theta)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("adaptive-threshold state must be numeric") from exc
        if not math.isfinite(v) or not math.isfinite(theta):
            raise ValueError("adaptive-threshold state must be finite")
        return v, theta

    def _validated_parameters(self) -> tuple[float, float, float, float, float, float, float]:
        """Return the finite numerical configuration without mutation."""
        try:
            v_rest = float(self.v_rest)
            v_reset = float(self.v_reset)
            theta_rest = float(self.theta_rest)
            delta_theta = float(self.delta_theta)
            tau_m = float(self.tau_m)
            tau_theta = float(self.tau_theta)
            dt = float(self.dt)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("adaptive-threshold parameters must be numeric") from exc
        values = (v_rest, v_reset, theta_rest, delta_theta, tau_m, tau_theta, dt)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("adaptive-threshold parameters must be finite")
        if delta_theta < 0.0:
            raise ValueError("delta_theta must be non-negative")
        if tau_m <= 0.0:
            raise ValueError("tau_m must be positive")
        if tau_theta <= 0.0:
            raise ValueError("tau_theta must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if theta_rest <= v_rest:
            raise ValueError("theta_rest must be greater than v_rest")
        if theta_rest <= v_reset:
            raise ValueError("theta_rest must be greater than v_reset")
        return values

    @staticmethod
    def _exact_relaxation(state: float, steady_state: float, tau: float, dt: float) -> float:
        """Return the exact constant-input relaxation over one ``dt`` interval."""
        try:
            decay = math.exp(-dt / tau)
        except OverflowError as exc:
            raise FloatingPointError("exact relaxation decay must be finite") from exc
        candidate = steady_state + (state - steady_state) * decay
        if not math.isfinite(candidate):
            raise FloatingPointError("exact relaxation candidate must be finite")
        return candidate

    def _candidate(self, current: float) -> tuple[float, float, int]:
        """Compute one validated candidate without mutating caller-visible state."""
        try:
            drive = float(current)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("current must be numeric") from exc
        if not math.isfinite(drive):
            raise ValueError("current must be finite")
        v, theta = self._validated_state()
        (
            v_rest,
            v_reset,
            theta_rest,
            delta_theta,
            tau_m,
            tau_theta,
            dt,
        ) = self._validated_parameters()
        next_v = self._exact_relaxation(v, v_rest + drive, tau_m, dt)
        next_theta = self._exact_relaxation(theta, theta_rest, tau_theta, dt)
        if next_v >= next_theta:
            spike_theta = next_theta + delta_theta
            if not math.isfinite(spike_theta):
                raise FloatingPointError("threshold jump candidate must be finite")
            return v_reset, spike_theta, 1
        return next_v, next_theta, 0

    def step(self, current: float = 0.0) -> int:
        """Advance one exact-relaxation interval and return a binary spike event.

        Mutation is atomic: invalid input, configuration, or candidate state
        leaves both dynamic states unchanged.
        """
        next_v, next_theta, spike = self._candidate(current)
        self.v, self.theta = next_v, next_theta
        return spike

    def simulate(
        self,
        current: npt.ArrayLike,
        *,
        backend: str = "auto",
    ) -> AdaptiveThresholdIFResult:
        """Run one atomic piecewise-constant-input batch on a maintained backend."""
        from sc_neurocore.accel.adaptive_threshold_if import simulate_adaptive_threshold_if

        result = simulate_adaptive_threshold_if(
            self.v,
            self.theta,
            self.v_rest,
            self.v_reset,
            self.theta_rest,
            self.delta_theta,
            self.tau_m,
            self.tau_theta,
            self.dt,
            current,
            backend=backend,
        )
        self.v = float(cast(float, result["v_final"]))
        self.theta = float(cast(float, result["theta_final"]))
        return result

    def reset(self) -> None:
        """Restore the documented rest state while preserving configuration."""
        self.v = self.v_rest
        self.theta = self.theta_rest


__all__ = ["AdaptiveThresholdIFNeuron", "AdaptiveThresholdIFResult"]
