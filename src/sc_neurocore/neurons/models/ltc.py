# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hasani et al. 2021 — liquid time-constant neuron

"""Liquid time-constant (LTC) neuron with input-modulated relaxation time."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

_exp = math.exp
_isfinite = math.isfinite
_tanh = math.tanh


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


def _safe_sigmoid(value: float) -> float:
    """Evaluate the logistic sigmoid without exponential overflow."""
    if value >= -700.0:
        return 1.0 / (1.0 + _exp(-value))
    exp_v = _exp(value)
    return exp_v / (1.0 + exp_v)


@dataclass
class LiquidTimeConstantNeuron:
    """Hasani et al. 2021 — liquid time-constant neuron.

    dx/dt = -(1/tau(x,I)) * x + (1/tau(x,I)) * f(x,I)
    where tau depends on input, making the neuron's time constant
    adaptive and input-driven.

    Reference: Hasani, R. et al. (2021). Proc. AAAI Conf. Artif. Intell. 35(9):7657–7666.

    Parameters
    ----------
    x : float, default=0.0
        Initial hidden membrane state.
    tau_base : float, default=10.0
        Baseline time constant in milliseconds; must be strictly positive.
    w_tau : float, default=-0.5
        Input weight modulating the liquid time-constant sigmoid.
    w_x : float, default=0.8
        Recurrent self-weight inside the target ``tanh``.
    w_in : float, default=1.0
        Input weight inside the target ``tanh``.
    bias : float, default=0.0
        Bias term inside the time-constant sigmoid.
    v_threshold : float, default=1.0
        Spiking threshold on ``x``.
    dt : float, default=1.0
        Discrete integration timestep in milliseconds; must be strictly positive.

    Raises
    ------
    TypeError
        If any parameter is a boolean or not a real number.
    ValueError
        If any parameter is non-finite, or if ``tau_base <= 0`` or ``dt <= 0``.
    """

    x: float = 0.0
    tau_base: float = 10.0
    w_tau: float = -0.5  # input→tau coupling
    w_x: float = 0.8
    w_in: float = 1.0
    bias: float = 0.0
    v_threshold: float = 1.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        """Validate initial state and parameters."""
        self.x = _require_finite_float(self.x, "x")
        self.tau_base = _require_positive_float(self.tau_base, "tau_base")
        self.w_tau = _require_finite_float(self.w_tau, "w_tau")
        self.w_x = _require_finite_float(self.w_x, "w_x")
        self.w_in = _require_finite_float(self.w_in, "w_in")
        self.bias = _require_finite_float(self.bias, "bias")
        self.v_threshold = _require_finite_float(self.v_threshold, "v_threshold")
        self.dt = _require_positive_float(self.dt, "dt")

    def step(self, current: float) -> int:
        """Advance the LTC hidden state by one Euler timestep.

        Parameters
        ----------
        current : float
            External input drive.

        Returns
        -------
        int
            ``1`` if the updated hidden state ``x`` reaches ``v_threshold``, else ``0``.

        Raises
        ------
        TypeError
            If ``current`` or any neuron field is a boolean or not a real number.
        ValueError
            If ``current`` or any field is non-finite, if ``tau_base <= 0`` or ``dt <= 0``,
            or if the candidate state ``x`` is non-finite; state is unchanged on error.
        """
        x = self.x
        tau_base = self.tau_base
        w_tau = self.w_tau
        w_x = self.w_x
        w_in = self.w_in
        bias = self.bias
        v_threshold = self.v_threshold
        dt = self.dt
        if not (
            current.__class__ is float
            and x.__class__ is float
            and tau_base.__class__ is float
            and w_tau.__class__ is float
            and w_x.__class__ is float
            and w_in.__class__ is float
            and bias.__class__ is float
            and v_threshold.__class__ is float
            and dt.__class__ is float
            and (current + x + tau_base + w_tau + w_x + w_in + bias + v_threshold + dt) * 0.0 == 0.0
            and tau_base > 0.0
            and dt > 0.0
        ):
            current = _require_finite_float(current, "current")
            x = _require_finite_float(x, "x")
            tau_base = _require_positive_float(tau_base, "tau_base")
            w_tau = _require_finite_float(w_tau, "w_tau")
            w_x = _require_finite_float(w_x, "w_x")
            w_in = _require_finite_float(w_in, "w_in")
            bias = _require_finite_float(bias, "bias")
            v_threshold = _require_finite_float(v_threshold, "v_threshold")
            dt = _require_positive_float(dt, "dt")
        tau = tau_base * _safe_sigmoid(w_tau * current + bias)
        if tau < 0.1:
            tau = 0.1
        f_target = _tanh(w_x * x + w_in * current)
        next_x = x + dt / tau * (-x + f_target)
        if not _isfinite(next_x):
            raise ValueError(f"updated hidden state x must be finite, got {next_x!r}.")
        if next_x >= v_threshold:
            self.x = 0.0
            return 1
        self.x = next_x
        return 0

    def reset(self) -> None:
        """Reset hidden state ``x`` to zero."""
        self.x = 0.0
