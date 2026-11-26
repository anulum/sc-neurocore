# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mensi et al. 2012 — Generalized IF with escape-rate spiking

"""Generalized integrate-and-fire population neuron with escape-rate spiking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import ClassVar

from sc_neurocore.neurons.seed_domain import NUMPY_SEED_DOMAIN, SeedDomain

_exp = math.exp
_isclose = math.isclose
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


def _require_int_seed(value: object, name: str) -> int:
    """Validate that ``value`` is an integer (excluding booleans) and return ``int``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    return int(value)


@dataclass
class _XorShift64:
    """Deterministic 64-bit xorshift* pseudo-random number generator."""

    state: int

    def __post_init__(self) -> None:
        if self.state == 0:
            self.state = 1

    def random(self) -> float:
        x = self.state & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 12) & 0xFFFFFFFFFFFFFFFF
        x ^= (x << 25) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 27) & 0xFFFFFFFFFFFFFFFF
        self.state = x & 0xFFFFFFFFFFFFFFFF
        sample = (self.state * 2685821657736338717) & 0xFFFFFFFFFFFFFFFF
        return float(sample >> 11) * (1.0 / 9007199254740992.0)


@dataclass
class GIFPopulationNeuron:
    """Generalized integrate-and-fire population neuron with escape-rate spiking.

    The deterministic subthreshold flow solves the coupled linear membrane and
    spike-history adaptation equations exactly over one fixed-current time step.
    Spiking then follows the Mensi et al. escape-rate hazard with a bounded
    Poisson interval probability.

    Reference: Mensi, S. et al. (2012). J. Neurophysiol. 107:1756-1775.

    Parameters
    ----------
    v : float, default=-65.0
        Initial membrane potential in millivolts.
    theta : float, default=-50.0
        Baseline escape-rate threshold in millivolts.
    eta : float, default=0.0
        Initial spike-history adaptation current.
    tau_m : float, default=20.0
        Membrane time constant in milliseconds; must be strictly positive.
    tau_eta : float, default=100.0
        Adaptation time constant in milliseconds; must be strictly positive.
    delta_v : float, default=2.0
        Escape-rate voltage sensitivity in millivolts; must be strictly positive.
    lambda_0 : float, default=0.001
        Baseline hazard rate at ``v == theta``; must be non-negative.
    eta_increment : float, default=5.0
        Adaptation current added after each emitted spike.
    v_rest : float, default=-65.0
        Resting membrane potential in millivolts.
    v_reset : float, default=-65.0
        Post-spike reset membrane potential in millivolts.
    dt : float, default=0.5
        Discrete integration timestep in milliseconds; must be strictly positive.
    seed : int, default=42
        Integer PRNG seed (negative integers are masked to 64 bits; the
        experiment contract enforces ``SEED_DOMAIN``).

    Raises
    ------
    TypeError
        If any floating-point field is a boolean or not a real number, or if
        ``seed`` is a boolean or not an integer.
    ValueError
        If any floating-point field is non-finite or violates its positivity
        or non-negativity domain.
    """

    #: Non-negative seed domain enforced by the experiment contract; the
    #: constructor accepts any integer seed and masks it to 64 bits.
    SEED_DOMAIN: ClassVar[SeedDomain] = NUMPY_SEED_DOMAIN

    v: float = -65.0
    theta: float = -50.0
    eta: float = 0.0
    tau_m: float = 20.0
    tau_eta: float = 100.0
    delta_v: float = 2.0
    lambda_0: float = 0.001
    eta_increment: float = 5.0
    v_rest: float = -65.0
    v_reset: float = -65.0
    dt: float = 0.5
    seed: int = 42
    _rng: _XorShift64 = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate initial state and parameters and initialise the PRNG."""
        self._validate_runtime()
        self.seed = _require_int_seed(self.seed, "seed")
        self._rng = _XorShift64(self.seed)

    def _validate_runtime(self) -> None:
        """Validate and normalise all floating-point state and parameter fields."""
        self.v = _require_finite_float(self.v, "v")
        self.theta = _require_finite_float(self.theta, "theta")
        self.eta = _require_finite_float(self.eta, "eta")
        self.tau_m = _require_positive_float(self.tau_m, "tau_m")
        self.tau_eta = _require_positive_float(self.tau_eta, "tau_eta")
        self.delta_v = _require_positive_float(self.delta_v, "delta_v")
        self.lambda_0 = _require_non_negative_float(self.lambda_0, "lambda_0")
        self.eta_increment = _require_finite_float(self.eta_increment, "eta_increment")
        self.v_rest = _require_finite_float(self.v_rest, "v_rest")
        self.v_reset = _require_finite_float(self.v_reset, "v_reset")
        self.dt = _require_positive_float(self.dt, "dt")

    def _advance_subthreshold(self, current: float) -> tuple[float, float]:
        eta_decay = _exp(-self.dt / self.tau_eta)
        membrane_decay = _exp(-self.dt / self.tau_m)
        x0 = self.v - self.v_rest - current
        eta_new = self.eta * eta_decay
        if _isclose(self.tau_m, self.tau_eta, rel_tol=1e-12, abs_tol=1e-12):
            x_new = membrane_decay * (x0 - (self.eta * self.dt / self.tau_m))
        else:
            coupling = self.tau_eta / (self.tau_eta - self.tau_m)
            x_new = (x0 * membrane_decay) - (self.eta * coupling * (eta_decay - membrane_decay))
        return self.v_rest + current + x_new, eta_new

    def _spike_probability(self, voltage: float) -> float:
        if self.lambda_0 == 0.0:
            return 0.0
        exponent = max(min((voltage - self.theta) / self.delta_v, 20.0), -745.0)
        hazard = self.lambda_0 * _exp(exponent)
        return min(max(1.0 - _exp(-hazard * self.dt), 0.0), 1.0)

    def step(self, current: float) -> int:
        """Advance the neuron by one exact subthreshold and escape-rate step.

        Parameters
        ----------
        current : float
            External injected current.

        Returns
        -------
        int
            ``1`` if an escape-rate spike is emitted, else ``0``.

        Raises
        ------
        TypeError
            If ``current`` or any runtime float attribute is a boolean or not
            a real number.
        ValueError
            If ``current`` or any attribute is non-finite or violates its
            domain, or if the subthreshold or post-spike update produces a
            non-finite state; membrane state, adaptation, and PRNG state are
            unchanged on error.
        """
        v = self.v
        theta = self.theta
        eta = self.eta
        tau_m = self.tau_m
        tau_eta = self.tau_eta
        delta_v = self.delta_v
        lambda_0 = self.lambda_0
        eta_increment = self.eta_increment
        v_rest = self.v_rest
        v_reset = self.v_reset
        dt = self.dt
        if not (
            current.__class__ is float
            and v.__class__ is float
            and theta.__class__ is float
            and eta.__class__ is float
            and tau_m.__class__ is float
            and tau_eta.__class__ is float
            and delta_v.__class__ is float
            and lambda_0.__class__ is float
            and eta_increment.__class__ is float
            and v_rest.__class__ is float
            and v_reset.__class__ is float
            and dt.__class__ is float
            and (
                current
                + v
                + theta
                + eta
                + tau_m
                + tau_eta
                + delta_v
                + lambda_0
                + eta_increment
                + v_rest
                + v_reset
                + dt
            )
            * 0.0
            == 0.0
            and tau_m > 0.0
            and tau_eta > 0.0
            and delta_v > 0.0
            and lambda_0 >= 0.0
            and dt > 0.0
        ):
            current = _require_finite_float(current, "current")
            self._validate_runtime()

        v_candidate, eta_candidate = self._advance_subthreshold(current)
        if not (v_candidate - v_candidate == 0.0 and eta_candidate - eta_candidate == 0.0):
            raise ValueError(
                f"subthreshold update produced non-finite state: v={v_candidate!r}, "
                f"eta={eta_candidate!r}."
            )

        rng_before = self._rng.state
        if self._rng.random() < self._spike_probability(v_candidate):
            spiked_eta = eta_candidate + self.eta_increment
            if not _isfinite(spiked_eta):
                self._rng.state = rng_before
                raise ValueError(f"post-spike adaptation state must be finite, got {spiked_eta!r}.")
            self.v = self.v_reset
            self.eta = spiked_eta
            return 1

        self.v = v_candidate
        self.eta = eta_candidate
        return 0

    def reset(self) -> None:
        """Restore membrane potential, adaptation, and PRNG to initial seeded state.

        Raises
        ------
        TypeError
            If ``v_rest`` is not a real number or ``seed`` is a boolean or not
            an integer.
        ValueError
            If ``v_rest`` is non-finite.
        """
        v_rest = _require_finite_float(self.v_rest, "v_rest")
        seed = _require_int_seed(self.seed, "seed")
        self.v = v_rest
        self.eta = 0.0
        self._rng = _XorShift64(seed)
