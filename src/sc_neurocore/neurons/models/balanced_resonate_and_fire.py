# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Balanced Resonate-and-Fire Neuron

"""Balanced Resonate-and-Fire neuron from Higuchi et al. (ICML 2024).

The implementation follows Algorithm 1 in "Balanced Resonate-and-Fire
Neurons": refractory threshold, smooth reset through the damping term, and the
discrete-time divergence boundary ``p(omega)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

_isfinite = math.isfinite
_sqrt = math.sqrt


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
        raise ValueError(f"{name} must be positive.")
    return result


def sustain_oscillation_boundary(omega: float, dt: float) -> float:
    """Return the BRF divergence boundary ``p(omega)``.

    ``p(omega) = (-1 + sqrt(1 - (dt * omega)^2)) / dt``.
    The value is real only when ``0 < dt * omega <= 1``.

    Parameters
    ----------
    omega : float
        Resonator angular frequency; must be finite and strictly positive.
    dt : float
        Discrete timestep; must be finite and strictly positive.

    Returns
    -------
    float
        Critical damping boundary ``p(omega) <= 0``.

    Raises
    ------
    TypeError
        If ``omega`` or ``dt`` is a boolean or not a real number.
    ValueError
        If ``omega`` or ``dt`` is non-finite or non-positive, or if ``dt * omega > 1``.
    """
    dt_val = _require_positive_float(dt, "dt")
    omega_val = _require_positive_float(omega, "omega")
    scaled = dt_val * omega_val
    if scaled > 1.0:
        raise ValueError(
            "BRF divergence boundary is undefined for dt * omega > 1; "
            f"got dt={dt_val!r}, omega={omega_val!r}."
        )
    return (-1.0 + _sqrt(max(0.0, 1.0 - scaled * scaled))) / dt_val


@dataclass
class BalancedResonateAndFireNeuron:
    """Balanced RF neuron with refractory threshold and smooth reset.

    State variables follow the paper notation ``u = x + i y`` and refractory
    state ``q``. One scalar update computes:

    ``b_t = p(omega) - b_offset - q_{t-1}``

    ``u_t = u_{t-1} + dt * ((b_t + i * omega) * u_{t-1} + current)``

    ``theta_t = theta_c + q_{t-1}``

    ``z_t = Heaviside(Re(u_t) - theta_t)``

    ``q_t = gamma * q_{t-1} + z_t``

    Reference: Higuchi, Kairat, Bohte, and Otte (2024), "Balanced
    Resonate-and-Fire Neurons", Proceedings of ICML 2024, Algorithm 1.

    Parameters
    ----------
    x : float, default=0.0
        Real component of the complex membrane state ``u``.
    y : float, default=0.0
        Imaginary component of the complex membrane state ``u``.
    q : float, default=0.0
        Refractory adaptation state.
    omega : float, default=10.0
        Angular frequency of oscillation; must satisfy ``0 < dt * omega <= 1``.
    b_offset : float, default=1.0
        Positive damping margin below ``p(omega)``.
    threshold : float, default=1.0
        Positive base firing threshold ``theta_c``.
    gamma : float, default=0.9
        Refractory decay factor in ``[0, 1)``.
    dt : float, default=0.01
        Positive integration timestep satisfying ``dt * omega <= 1``.

    Raises
    ------
    TypeError
        If any parameter is a boolean or not a real number.
    ValueError
        If any parameter is non-finite or violates its domain or stability bound.
    """

    x: float = 0.0
    y: float = 0.0
    q: float = 0.0
    omega: float = 10.0
    b_offset: float = 1.0
    threshold: float = 1.0
    gamma: float = 0.9
    dt: float = 0.01

    def __post_init__(self) -> None:
        """Validate and normalise initial state and parameters."""
        (
            self.x,
            self.y,
            self.q,
            self.omega,
            self.b_offset,
            self.threshold,
            self.gamma,
            self.dt,
        ) = self._validate_parameters()

    @property
    def p_omega(self) -> float:
        """Current divergence boundary for ``omega`` and ``dt``."""
        return sustain_oscillation_boundary(self.omega, self.dt)

    @property
    def damping(self) -> float:
        """Current smooth-reset damping ``b_t`` before the next step.

        Returns
        -------
        float
            Effective damping ``p(omega) - b_offset - q``.

        Raises
        ------
        TypeError
            If ``omega``, ``dt``, ``b_offset``, or ``q`` is a boolean or non-real.
        ValueError
            If any input parameter is invalid or the computed damping overflows.
        """
        p_val = self.p_omega
        b_offset = _require_positive_float(self.b_offset, "b_offset")
        q = _require_finite_float(self.q, "q")
        result = p_val - b_offset - q
        if not _isfinite(result):
            raise ValueError(f"damping must be finite, got {result!r}.")
        return result

    @property
    def dynamic_threshold(self) -> float:
        """Current refractory threshold ``theta_c + q``.

        Returns
        -------
        float
            Effective firing threshold ``threshold + q``.

        Raises
        ------
        TypeError
            If ``threshold`` or ``q`` is a boolean or non-real.
        ValueError
            If any input parameter is invalid or the computed threshold overflows.
        """
        threshold = _require_positive_float(self.threshold, "threshold")
        q = _require_finite_float(self.q, "q")
        result = threshold + q
        if not _isfinite(result):
            raise ValueError(f"dynamic_threshold must be finite, got {result!r}.")
        return result

    def step(self, current: float) -> int:
        """Advance one BRF timestep and return the binary spike ``z_t``.

        Parameters
        ----------
        current : float
            External input drive added to the real membrane derivative.

        Returns
        -------
        int
            ``1`` if ``Re(u_t) >= theta_t``, else ``0``.

        Raises
        ------
        TypeError
            If ``current`` or any neuron field is a boolean or not a real number.
        ValueError
            If ``current`` or any field is non-finite or out of domain, or if any
            candidate state or threshold overflows; state is unchanged on error.
        """
        x = self.x
        y = self.y
        q = self.q
        omega = self.omega
        b_offset = self.b_offset
        threshold = self.threshold
        gamma = self.gamma
        dt = self.dt
        if not (
            current.__class__ is float
            and x.__class__ is float
            and y.__class__ is float
            and q.__class__ is float
            and omega.__class__ is float
            and b_offset.__class__ is float
            and threshold.__class__ is float
            and gamma.__class__ is float
            and dt.__class__ is float
            and (current + x + y + q + omega + b_offset + threshold + gamma + dt) * 0.0 == 0.0
            and omega > 0.0
            and b_offset > 0.0
            and threshold > 0.0
            and 0.0 <= gamma < 1.0
            and dt > 0.0
            and dt * omega <= 1.0
        ):
            x, y, q, omega, b_offset, threshold, gamma, dt = self._validate_parameters()
            current = _require_finite_float(current, "current")
        scaled = dt * omega
        disc = 1.0 - scaled * scaled
        p_omega = (-1.0 + _sqrt(disc if disc > 0.0 else 0.0)) / dt
        b_t = p_omega - b_offset - q
        theta_t = threshold + q
        next_x = x + dt * (b_t * x - omega * y + current)
        next_y = y + dt * (omega * x + b_t * y)
        if not (
            b_t - b_t == 0.0
            and theta_t - theta_t == 0.0
            and next_x - next_x == 0.0
            and next_y - next_y == 0.0
        ):
            raise ValueError("BRF step produced a non-finite state or threshold.")

        if next_x >= theta_t:
            self.x = next_x
            self.y = next_y
            self.q = gamma * q + 1.0
            return 1
        self.x = next_x
        self.y = next_y
        self.q = gamma * q
        return 0

    def reset(self) -> None:
        """Reset membrane and refractory state to rest."""
        self.x = 0.0
        self.y = 0.0
        self.q = 0.0

    def state(self) -> dict[str, float]:
        """Return a compact state snapshot for reproducibility tests.

        Returns
        -------
        dict[str, float]
            Dictionary containing state variables, parameters, and derived quantities.

        Raises
        ------
        TypeError
            If any state or parameter field is a boolean or not a real number.
        ValueError
            If any state, parameter, or derived quantity is invalid or non-finite.
        """
        x, y, q, omega, b_offset, threshold, gamma, dt = self._validate_parameters()
        return {
            "x": x,
            "y": y,
            "q": q,
            "omega": omega,
            "b_offset": b_offset,
            "threshold": threshold,
            "gamma": gamma,
            "dt": dt,
            "p_omega": self.p_omega,
            "damping": self.damping,
            "dynamic_threshold": self.dynamic_threshold,
        }

    def _validate_parameters(
        self,
    ) -> tuple[float, float, float, float, float, float, float, float]:
        """Validate all state and parameter fields and return normalised floats."""
        dt = _require_positive_float(self.dt, "dt")
        omega = _require_positive_float(self.omega, "omega")
        if dt * omega > 1.0:
            raise ValueError(
                "BRF requires dt * omega <= 1 so p(omega) remains real; "
                f"got dt={dt!r}, omega={omega!r}."
            )
        b_offset = _require_positive_float(self.b_offset, "b_offset")
        threshold = _require_positive_float(self.threshold, "threshold")
        gamma = _require_finite_float(self.gamma, "gamma")
        if not 0.0 <= gamma < 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma < 1.")
        x = _require_finite_float(self.x, "x")
        y = _require_finite_float(self.y, "y")
        q = _require_finite_float(self.q, "q")
        return x, y, q, omega, b_offset, threshold, gamma, dt
