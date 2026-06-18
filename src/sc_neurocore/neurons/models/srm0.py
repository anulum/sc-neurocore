# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SRM0 (Spike Response Model, zeroth order)

from __future__ import annotations

from dataclasses import dataclass
import math


def _finite_scalar(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real finite scalar")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


@dataclass
class SRM0Neuron:
    """Spike Response Model with exact constant-current kernel flow.

    The zeroth-order Spike Response Model keeps membrane potential ``v`` and a
    refractory afterhyperpolarisation kernel ``eta``. During one step, external
    current is held constant and the coupled linear system for ``(v, eta)`` is
    integrated analytically. This avoids the first-order membrane Euler error
    while preserving the SRM0 threshold/reset semantics.

    Parameters
    ----------
    v:
        Membrane potential state.
    v_rest:
        Rest potential and post-spike voltage reset.
    v_threshold:
        Spike threshold. A spike is emitted when the exact-flow candidate
        reaches or exceeds this value.
    tau_m:
        Membrane time constant. Must be positive and finite.
    tau_eta:
        Refractory-kernel decay time constant. Must be positive and finite.
    eta_reset:
        Positive refractory-kernel amplitude. On spike, ``eta`` is set to
        ``-eta_reset``.
    resistance:
        Current-to-voltage gain.
    dt:
        Integration step. Must be positive and finite.

    Raises
    ------
    TypeError
        If a scalar parameter is not numeric.
    ValueError
        If a parameter is non-finite or a positive contract is violated.

    References
    ----------
    Gerstner, W., & Kistler, W. M. (2002). Spiking Neuron Models:
    Single Neurons, Populations, Plasticity. Cambridge University Press,
    chapter 4.
    """

    v: float = 0.0
    v_rest: float = 0.0
    v_threshold: float = 1.0
    tau_m: float = 20.0
    tau_eta: float = 50.0
    eta_reset: float = 5.0
    resistance: float = 1.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        """Initialise private kernel state after validating public parameters."""

        self.v = _finite_scalar("v", self.v)
        self.v_rest = _finite_scalar("v_rest", self.v_rest)
        self.v_threshold = _finite_scalar("v_threshold", self.v_threshold)
        self.tau_m = _finite_scalar("tau_m", self.tau_m)
        self.tau_eta = _finite_scalar("tau_eta", self.tau_eta)
        self.eta_reset = _finite_scalar("eta_reset", self.eta_reset)
        self.resistance = _finite_scalar("resistance", self.resistance)
        self.dt = _finite_scalar("dt", self.dt)
        if self.tau_m <= 0.0:
            raise ValueError("tau_m must be positive")
        if self.tau_eta <= 0.0:
            raise ValueError("tau_eta must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.eta_reset < 0.0:
            raise ValueError("eta_reset must be non-negative")
        self._eta = 0.0
        self._last_spike_time = -1000.0
        self._t = 0.0

    def step(self, current: float) -> int:
        """Advance one exact-flow SRM0 step.

        Parameters
        ----------
        current:
            External current held constant during the step.

        Returns
        -------
        int
            ``1`` when the exact membrane candidate reaches threshold,
            otherwise ``0``.

        Raises
        ------
        ValueError
            If runtime state, current, or the exact-flow candidate is invalid.
            State is preserved on failure.
        """

        current = _finite_scalar("current", current)
        self._validate_runtime_state()
        next_v, next_eta = self._exact_candidate(current)
        if not (math.isfinite(next_v) and math.isfinite(next_eta)):
            raise ValueError("SRM0 exact-flow candidate must be finite")

        next_t = self._t + self.dt
        if next_v >= self.v_threshold:
            self.v = self.v_rest
            self._eta = -self.eta_reset
            self._t = next_t
            self._last_spike_time = next_t
            return 1
        self.v = next_v
        self._eta = next_eta
        self._t = next_t
        return 0

    def reset(self) -> None:
        """Restore voltage, refractory kernel, and internal clock state."""

        self.v = self.v_rest
        self._eta = 0.0
        self._t = 0.0
        self._last_spike_time = -1000.0

    def get_state(self) -> dict[str, float]:
        """Return the current diagnostic SRM0 state."""

        return {"v": self.v, "eta": self._eta, "t": self._t}

    def _validate_runtime_state(self) -> None:
        _finite_scalar("v", self.v)
        _finite_scalar("eta", self._eta)
        _finite_scalar("t", self._t)
        _finite_scalar("last_spike_time", self._last_spike_time)
        if self.tau_m <= 0.0 or self.tau_eta <= 0.0 or self.dt <= 0.0:
            raise ValueError("SRM0 positive time constants and dt are required")

    def _eta_coupling_integral(self) -> float:
        membrane_decay = math.exp(-self.dt / self.tau_m)
        eta_decay = math.exp(-self.dt / self.tau_eta)
        rate_delta = (1.0 / self.tau_m) - (1.0 / self.tau_eta)
        if abs(rate_delta) < 1.0e-14:
            return self.dt * membrane_decay / self.tau_m
        return (eta_decay - membrane_decay) / (self.tau_m * rate_delta)

    def _exact_candidate(self, current: float) -> tuple[float, float]:
        membrane_decay = math.exp(-self.dt / self.tau_m)
        eta_decay = math.exp(-self.dt / self.tau_eta)
        steady = self.v_rest + self.resistance * current
        next_eta = self._eta * eta_decay
        next_v = (
            steady + (self.v - steady) * membrane_decay + self._eta * self._eta_coupling_integral()
        )
        return next_v, next_eta
