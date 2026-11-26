# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Zenke & Ganguli 2018 — LIF with SuperSpike surrogate

"""SuperSpike LIF neuron with Van Rossum eligibility trace and surrogate gradient."""

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


def _require_non_negative_float(value: object, name: str) -> float:
    """Validate that ``value`` is a non-negative finite real number."""
    result = _require_finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")
    return result


def _surrogate_grad_value(v: float, v_threshold: float, beta_sg: float) -> float:
    """Evaluate ``1 / (beta_sg * |v - v_threshold| + 1)^2`` without squaring overflow."""
    if beta_sg == 0.0:
        return 1.0
    denom = beta_sg * abs(v - v_threshold) + 1.0
    if denom > 1e154:
        inv = 1.0 / denom
        return inv * inv
    return 1.0 / (denom * denom)


@dataclass
class SuperSpikeNeuron:
    """Zenke & Ganguli 2018 — LIF with SuperSpike surrogate gradient.

    Uses Van Rossum filtered eligibility traces and a smooth surrogate
    gradient sigma'(V) = 1/(beta * |V - V_th| + 1)^2.

    Reference: Zenke, F. & Ganguli, S. (2018). Neural Comput. 30:1514–1541.

    Parameters
    ----------
    v : float, default=0.0
        Initial membrane potential.
    trace : float, default=0.0
        Initial Van Rossum eligibility trace.
    tau_m : float, default=10.0
        Membrane time constant in milliseconds; must be strictly positive.
    tau_e : float, default=10.0
        Eligibility filter time constant in milliseconds; must be strictly positive.
    v_threshold : float, default=1.0
        Firing threshold.
    v_reset : float, default=0.0
        Post-spike reset potential.
    beta_sg : float, default=10.0
        Surrogate-gradient sharpness; must be non-negative.
    dt : float, default=1.0
        Discrete integration timestep in milliseconds; must be strictly positive.

    Raises
    ------
    TypeError
        If any parameter is a boolean or not a real number.
    ValueError
        If any parameter is non-finite, if ``tau_m <= 0``, ``tau_e <= 0``, or ``dt <= 0``,
        or if ``beta_sg < 0``.
    """

    v: float = 0.0
    trace: float = 0.0  # Van Rossum eligibility trace
    tau_m: float = 10.0
    tau_e: float = 10.0  # eligibility filter time constant
    v_threshold: float = 1.0
    v_reset: float = 0.0
    beta_sg: float = 10.0  # surrogate gradient sharpness
    dt: float = 1.0
    alpha_m: float = field(init=False)
    alpha_e: float = field(init=False)

    def __post_init__(self) -> None:
        """Validate initial fields and precompute membrane and trace decay factors."""
        self.v = _require_finite_float(self.v, "v")
        self.trace = _require_finite_float(self.trace, "trace")
        self.tau_m = _require_positive_float(self.tau_m, "tau_m")
        self.tau_e = _require_positive_float(self.tau_e, "tau_e")
        self.v_threshold = _require_finite_float(self.v_threshold, "v_threshold")
        self.v_reset = _require_finite_float(self.v_reset, "v_reset")
        self.beta_sg = _require_non_negative_float(self.beta_sg, "beta_sg")
        self.dt = _require_positive_float(self.dt, "dt")
        self.alpha_m = math.exp(-self.dt / self.tau_m)
        self.alpha_e = math.exp(-self.dt / self.tau_e)

    def surrogate_grad(self) -> float:
        """Compute the SuperSpike fast-sigmoid surrogate derivative at current ``v``.

        Returns
        -------
        float
            Value of ``1 / (beta_sg * |v - v_threshold| + 1)^2`` in ``[0, 1]``.

        Raises
        ------
        TypeError
            If ``v``, ``v_threshold``, or ``beta_sg`` is a boolean or not a real number.
        ValueError
            If ``v``, ``v_threshold``, or ``beta_sg`` is non-finite, or if ``beta_sg < 0``.
        """
        v = _require_finite_float(self.v, "v")
        v_threshold = _require_finite_float(self.v_threshold, "v_threshold")
        beta_sg = _require_non_negative_float(self.beta_sg, "beta_sg")
        return _surrogate_grad_value(v, v_threshold, beta_sg)

    def step(self, current: float) -> int:
        """Advance membrane potential and eligibility trace by one timestep.

        Parameters
        ----------
        current : float
            External input drive.

        Returns
        -------
        int
            ``1`` if the updated membrane potential reaches ``v_threshold``, else ``0``.

        Raises
        ------
        TypeError
            If ``current`` or any neuron field is a boolean or not a real number.
        ValueError
            If ``current`` or any field is non-finite, violates its positivity or
            non-negativity domain, or if candidate ``v`` or ``trace`` overflows;
            state is unchanged on error.
        """
        v = self.v
        trace = self.trace
        tau_m = self.tau_m
        tau_e = self.tau_e
        v_threshold = self.v_threshold
        v_reset = self.v_reset
        beta_sg = self.beta_sg
        dt = self.dt
        alpha_m = self.alpha_m
        alpha_e = self.alpha_e
        if not (
            current.__class__ is float
            and v.__class__ is float
            and trace.__class__ is float
            and tau_m.__class__ is float
            and tau_e.__class__ is float
            and v_threshold.__class__ is float
            and v_reset.__class__ is float
            and beta_sg.__class__ is float
            and dt.__class__ is float
            and alpha_m.__class__ is float
            and alpha_e.__class__ is float
            and (
                current
                + v
                + trace
                + tau_m
                + tau_e
                + v_threshold
                + v_reset
                + beta_sg
                + dt
                + alpha_m
                + alpha_e
            )
            * 0.0
            == 0.0
            and tau_m > 0.0
            and tau_e > 0.0
            and beta_sg >= 0.0
            and dt > 0.0
        ):
            current = _require_finite_float(current, "current")
            v = _require_finite_float(v, "v")
            trace = _require_finite_float(trace, "trace")
            _require_positive_float(tau_m, "tau_m")
            _require_positive_float(tau_e, "tau_e")
            v_threshold = _require_finite_float(v_threshold, "v_threshold")
            v_reset = _require_finite_float(v_reset, "v_reset")
            beta_sg = _require_non_negative_float(beta_sg, "beta_sg")
            _require_positive_float(dt, "dt")
            alpha_m = _require_finite_float(alpha_m, "alpha_m")
            alpha_e = _require_finite_float(alpha_e, "alpha_e")
        next_v = alpha_m * v + current
        if not _isfinite(next_v):
            raise ValueError(f"updated membrane potential v must be finite, got {next_v!r}.")
        sg = _surrogate_grad_value(next_v, v_threshold, beta_sg)
        next_trace = alpha_e * trace + sg
        if not _isfinite(next_trace):
            raise ValueError(f"updated eligibility trace must be finite, got {next_trace!r}.")
        self.trace = next_trace
        if next_v >= v_threshold:
            self.v = v_reset
            return 1
        self.v = next_v
        return 0

    def reset(self) -> None:
        """Reset membrane potential and eligibility trace to zero."""
        self.v, self.trace = 0.0, 0.0
