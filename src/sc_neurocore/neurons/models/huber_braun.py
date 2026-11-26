# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Braun, Huber et al. 1998 — cold receptor,

"""Huber-Braun cold-receptor conductance neuron with slow ionic gates and noise."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, ClassVar

import numpy as np

from sc_neurocore.neurons.seed_domain import NUMPY_SEED_DOMAIN, SeedDomain

_exp = math.exp
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


def _safe_sigmoid(value: float) -> float:
    """Evaluate the logistic sigmoid without exponential overflow."""
    if value >= -700.0:
        return 1.0 / (1.0 + _exp(-value))
    exp_v = _exp(value)
    return exp_v / (1.0 + exp_v)


@dataclass
class HuberBraunNeuron:
    """Braun, Huber et al. 1998 — cold receptor, temperature-dependent.

    4 ODEs: V, a_sd (slow depolarizing), a_sr (slow repolarizing), a_r.

    Reference: Braun, H.A. et al. (1998). Int. J. Bifurcation Chaos 8:881–889.

    Parameters
    ----------
    v : float, default=-50.0
        Initial membrane potential in millivolts.
    a_sd : float, default=0.0
        Initial slow depolarising activation variable.
    a_sr : float, default=0.0
        Initial slow repolarising activation variable.
    g_sd : float, default=1.5
        Slow depolarising maximal conductance; must be non-negative.
    g_sr : float, default=0.4
        Slow repolarising maximal conductance; must be non-negative.
    g_l : float, default=0.1
        Leak conductance; must be non-negative.
    e_sd : float, default=50.0
        Slow depolarising reversal potential in millivolts.
    e_sr : float, default=-90.0
        Slow repolarising reversal potential in millivolts.
    e_l : float, default=-60.0
        Leak reversal potential in millivolts.
    tau_sd : float, default=10.0
        Slow depolarising time constant in milliseconds; must be strictly positive.
    tau_sr : float, default=20.0
        Slow repolarising time constant in milliseconds; must be strictly positive.
    eta : float, default=0.012
        Gaussian membrane noise amplitude; must be non-negative.
    dt : float, default=0.1
        Discrete integration timestep in milliseconds; must be strictly positive.
    v_threshold : float, default=-20.0
        Upward spike-detection threshold in millivolts.
    seed : int or None, default=None
        Seed of the neuron's own noise generator, any non-negative integer (NumPy's
        included, stored as a Python ``int``) up to ``SEED_DOMAIN``'s bound. With a seed the
        noise comes from that ``numpy.random.Generator``, two neurons with the same seed give
        the same trajectory whatever else draws random numbers, and ``reset`` restarts the
        stream. Without one the noise comes from ``np.random.randn``, as before, so
        ``np.random.seed`` governs it.

    Raises
    ------
    TypeError
        If any parameter is a boolean or not a real number, or ``seed`` is a boolean or
        not an integer.
    ValueError
        If any parameter is non-finite or violates its positivity or
        non-negativity domain.
    """

    v: float = -50.0
    a_sd: float = 0.0
    a_sr: float = 0.0
    g_sd: float = 1.5
    g_sr: float = 0.4
    g_l: float = 0.1
    e_sd: float = 50.0
    e_sr: float = -90.0
    e_l: float = -60.0
    tau_sd: float = 10.0
    tau_sr: float = 20.0
    eta: float = 0.012
    dt: float = 0.1
    v_threshold: float = -20.0
    seed: int | np.integer[Any] | None = None
    _rng: np.random.Generator | None = field(default=None, init=False, repr=False, compare=False)

    SEED_DOMAIN: ClassVar[SeedDomain] = NUMPY_SEED_DOMAIN

    def __post_init__(self) -> None:
        """Validate and normalise all initial state and parameter fields."""
        self.v = _require_finite_float(self.v, "v")
        self.a_sd = _require_finite_float(self.a_sd, "a_sd")
        self.a_sr = _require_finite_float(self.a_sr, "a_sr")
        self.g_sd = _require_non_negative_float(self.g_sd, "g_sd")
        self.g_sr = _require_non_negative_float(self.g_sr, "g_sr")
        self.g_l = _require_non_negative_float(self.g_l, "g_l")
        self.e_sd = _require_finite_float(self.e_sd, "e_sd")
        self.e_sr = _require_finite_float(self.e_sr, "e_sr")
        self.e_l = _require_finite_float(self.e_l, "e_l")
        self.tau_sd = _require_positive_float(self.tau_sd, "tau_sd")
        self.tau_sr = _require_positive_float(self.tau_sr, "tau_sr")
        self.eta = _require_non_negative_float(self.eta, "eta")
        self.dt = _require_positive_float(self.dt, "dt")
        self.v_threshold = _require_finite_float(self.v_threshold, "v_threshold")
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
                raise TypeError("seed must be an integer or None")
            seed = int(self.seed)
            if seed < 0:
                raise ValueError("seed must be non-negative")
            self.seed = seed
        self._rng = None if self.seed is None else np.random.default_rng(self.seed)

    def step(self, current: float) -> int:
        """Advance slow gates and membrane potential by one Euler timestep.

        Parameters
        ----------
        current : float
            External injected current.

        Returns
        -------
        int
            ``1`` on an upward crossing of ``v_threshold``, else ``0``.

        Raises
        ------
        TypeError
            If ``current`` or any neuron field is a boolean or not a real number.
        ValueError
            If ``current`` or any field is non-finite or out of domain, or if
            candidate state variables overflow; state is unchanged on error.
        """
        v = self.v
        a_sd = self.a_sd
        a_sr = self.a_sr
        g_sd = self.g_sd
        g_sr = self.g_sr
        g_l = self.g_l
        e_sd = self.e_sd
        e_sr = self.e_sr
        e_l = self.e_l
        tau_sd = self.tau_sd
        tau_sr = self.tau_sr
        eta = self.eta
        dt = self.dt
        v_threshold = self.v_threshold
        if not (
            current.__class__ is float
            and v.__class__ is float
            and a_sd.__class__ is float
            and a_sr.__class__ is float
            and g_sd.__class__ is float
            and g_sr.__class__ is float
            and g_l.__class__ is float
            and e_sd.__class__ is float
            and e_sr.__class__ is float
            and e_l.__class__ is float
            and tau_sd.__class__ is float
            and tau_sr.__class__ is float
            and eta.__class__ is float
            and dt.__class__ is float
            and v_threshold.__class__ is float
            and (
                current
                + v
                + a_sd
                + a_sr
                + g_sd
                + g_sr
                + g_l
                + e_sd
                + e_sr
                + e_l
                + tau_sd
                + tau_sr
                + eta
                + dt
                + v_threshold
            )
            * 0.0
            == 0.0
            and g_sd >= 0.0
            and g_sr >= 0.0
            and g_l >= 0.0
            and tau_sd > 0.0
            and tau_sr > 0.0
            and eta >= 0.0
            and dt > 0.0
        ):
            current = _require_finite_float(current, "current")
            v = _require_finite_float(v, "v")
            a_sd = _require_finite_float(a_sd, "a_sd")
            a_sr = _require_finite_float(a_sr, "a_sr")
            g_sd = _require_non_negative_float(g_sd, "g_sd")
            g_sr = _require_non_negative_float(g_sr, "g_sr")
            g_l = _require_non_negative_float(g_l, "g_l")
            e_sd = _require_finite_float(e_sd, "e_sd")
            e_sr = _require_finite_float(e_sr, "e_sr")
            e_l = _require_finite_float(e_l, "e_l")
            tau_sd = _require_positive_float(tau_sd, "tau_sd")
            tau_sr = _require_positive_float(tau_sr, "tau_sr")
            eta = _require_non_negative_float(eta, "eta")
            dt = _require_positive_float(dt, "dt")
            v_threshold = _require_finite_float(v_threshold, "v_threshold")

        arg = (v + 40.0) / 6.0
        sd_inf = _safe_sigmoid(arg)
        sr_inf = _safe_sigmoid(-arg)
        next_a_sd = a_sd + (sd_inf - a_sd) / tau_sd * dt
        next_a_sr = a_sr + (sr_inf - a_sr) / tau_sr * dt
        i_sd = g_sd * next_a_sd * (v - e_sd)
        i_sr = g_sr * next_a_sr * (v - e_sr)
        i_l = g_l * (v - e_l)
        rng = self._rng
        if rng is None:
            noise = float(np.random.randn())
        else:
            # A seeded neuron's generator is part of its state: a refused step leaves it unmoved.
            drawn_from = rng.bit_generator.state
            noise = float(rng.standard_normal())
        next_v = v + (-i_sd - i_sr - i_l + current + eta * noise) * dt
        if not (
            next_a_sd - next_a_sd == 0.0 and next_a_sr - next_a_sr == 0.0 and next_v - next_v == 0.0
        ):
            if rng is not None:
                rng.bit_generator.state = drawn_from
            raise ValueError("HuberBraun step produced a non-finite state.")

        self.a_sd = next_a_sd
        self.a_sr = next_a_sr
        self.v = next_v
        return 1 if (next_v >= v_threshold and v < v_threshold) else 0

    def reset(self) -> None:
        """Reset membrane potential and slow gating variables; a seeded neuron restarts its noise."""
        self.v = -50.0
        self.a_sd, self.a_sr = 0.0, 0.0
        if self.seed is not None:
            self._rng = np.random.default_rng(self.seed)
