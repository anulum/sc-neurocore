# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Motor Unit (Alpha Motor Neuron + Muscle Fibre)

"""Motor unit: an LIF alpha motor neuron driving a first-order muscle twitch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

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


def _require_bounded_float(value: object, name: str, low: float, high: float) -> float:
    """Validate that ``value`` is finite and within ``[low, high]``."""
    result = _require_finite_float(value, name)
    if result < low or result > high:
        raise ValueError(f"{name} must be within [{low}, {high}], got {value!r}.")
    return result


def _relax(previous: float, steady: float, tau: float, dt: float) -> float:
    return steady + (previous - steady) * _exp(-dt / tau)


def _valid_voltage(value: float) -> bool:
    return _isfinite(value) and -150.0 <= value <= 100.0


@dataclass
class MotorUnit:
    """Motor unit — alpha motor neuron + muscle fibre.

    Each spike triggers a muscle twitch. Force output is summation of
    overlapping twitches (rate coding). The twitch is first-order: each
    spike adds ``twitch_amp`` to the force (capped at 1), which then decays
    with ``tau_twitch``.

    Reference: Fuglevand et al. (1993) J Neurophysiol 70(6);
    Heckman & Enoka (2012) Compr Physiol 2(4).

    Parameters
    ----------
    v : float, default=-65.0
        Initial membrane potential in millivolts within ``[-150, 100]``.
    v_rest : float, default=-65.0
        Resting potential in millivolts within ``[-150, 100]``.
    v_reset : float, default=-70.0
        Post-spike reset potential in millivolts within ``[-150, 100]``; must
        satisfy ``v_reset < v_threshold``.
    v_threshold : float, default=-50.0
        Spike threshold in millivolts within ``[-150, 100]``.
    tau_m : float, default=10.0
        Membrane time constant in milliseconds; must be strictly positive.
    adapt : float, default=0.0
        Initial spike-frequency adaptation state.
    tau_adapt : float, default=100.0
        Adaptation time constant in milliseconds; must be strictly positive.
    a_adapt : float, default=0.2
        Subthreshold adaptation coupling coefficient.
    gain : float, default=1.0
        Descending drive gain; must be non-negative.
    force : float, default=0.0
        Initial normalised muscle force within ``[0, 1]``.
    twitch_amp : float, default=0.05
        Force increment added on each spike; must be non-negative.
    tau_twitch : float, default=90.0
        Muscle twitch decay time constant in milliseconds; must be strictly positive.
    force_decay : float, default=0.0
        Unused compatibility field retained for descriptor schema parity.
    dt : float, default=0.5
        Integration timestep in milliseconds; must be strictly positive.

    Raises
    ------
    TypeError
        If any parameter is a boolean or not a real number.
    ValueError
        If any parameter is non-finite, outside its domain, or violates
        ``v_reset < v_threshold``.
    """

    v: float = -65.0
    v_rest: float = -65.0
    v_reset: float = -70.0
    v_threshold: float = -50.0
    tau_m: float = 10.0
    adapt: float = 0.0
    tau_adapt: float = 100.0
    a_adapt: float = 0.2
    gain: float = 1.0
    force: float = 0.0
    twitch_amp: float = 0.05
    tau_twitch: float = 90.0
    force_decay: float = 0.0
    dt: float = 0.5

    def __post_init__(self) -> None:
        """Validate and normalise all initial state variables and parameters."""
        self._validate_runtime()

    def _validate_runtime(self) -> None:
        """Validate and normalise every state variable and parameter field."""
        self.v = _require_bounded_float(self.v, "v", -150.0, 100.0)
        self.v_rest = _require_bounded_float(self.v_rest, "v_rest", -150.0, 100.0)
        self.v_reset = _require_bounded_float(self.v_reset, "v_reset", -150.0, 100.0)
        self.v_threshold = _require_bounded_float(self.v_threshold, "v_threshold", -150.0, 100.0)
        if self.v_reset >= self.v_threshold:
            raise ValueError(
                f"v_reset ({self.v_reset!r}) must be strictly less than "
                f"v_threshold ({self.v_threshold!r})."
            )
        self.tau_m = _require_positive_float(self.tau_m, "tau_m")
        self.adapt = _require_finite_float(self.adapt, "adapt")
        self.tau_adapt = _require_positive_float(self.tau_adapt, "tau_adapt")
        self.a_adapt = _require_finite_float(self.a_adapt, "a_adapt")
        self.gain = _require_non_negative_float(self.gain, "gain")
        self.force = _require_bounded_float(self.force, "force", 0.0, 1.0)
        self.twitch_amp = _require_non_negative_float(self.twitch_amp, "twitch_amp")
        self.tau_twitch = _require_positive_float(self.tau_twitch, "tau_twitch")
        self.force_decay = _require_finite_float(self.force_decay, "force_decay")
        self.dt = _require_positive_float(self.dt, "dt")

    @classmethod
    def slow(cls) -> MotorUnit:
        """Slow motor unit (type S): small, fatigue-resistant, low force."""
        return cls()

    @classmethod
    def fast(cls) -> MotorUnit:
        """Fast motor unit (type FF): large, fatigable, high force."""
        return cls(
            tau_m=6.0,
            tau_adapt=50.0,
            a_adapt=0.1,
            twitch_amp=0.3,
            tau_twitch=30.0,
        )

    def step(self, drive: float = 0.0) -> int:
        """Advance one step with the descending drive.

        The force first decays by ``exp(-dt / tau_twitch)``. ``v`` relaxes exactly
        towards ``v_rest + gain * max(drive, 0) - adapt`` with ``tau_m``; then
        ``adapt`` relaxes towards ``a_adapt * (v - v_rest)`` of that new ``v``,
        before any reset, with ``tau_adapt``. Reaching ``v_threshold`` sets ``v``
        to ``v_reset`` and adds ``twitch_amp`` to the force, capped at 1.

        Parameters
        ----------
        drive : float, optional
            Descending drive; negative values count as 0. Default 0.

        Returns
        -------
        int
            ``1`` on a spike, else ``0``.

        Raises
        ------
        TypeError
            If ``drive`` or any neuron field is a boolean or not a real number.
        ValueError
            If ``drive`` or any field is non-finite or violates its domain, or
            if the candidate voltage leaves ``[-150, 100]`` mV or the candidate
            adaptation state becomes non-finite; state is unchanged on error.
        """
        v = self.v
        v_rest = self.v_rest
        v_reset = self.v_reset
        v_threshold = self.v_threshold
        tau_m = self.tau_m
        adapt = self.adapt
        tau_adapt = self.tau_adapt
        a_adapt = self.a_adapt
        gain = self.gain
        force = self.force
        twitch_amp = self.twitch_amp
        tau_twitch = self.tau_twitch
        force_decay = self.force_decay
        dt = self.dt
        if not (
            drive.__class__ is float
            and v.__class__ is float
            and v_rest.__class__ is float
            and v_reset.__class__ is float
            and v_threshold.__class__ is float
            and tau_m.__class__ is float
            and adapt.__class__ is float
            and tau_adapt.__class__ is float
            and a_adapt.__class__ is float
            and gain.__class__ is float
            and force.__class__ is float
            and twitch_amp.__class__ is float
            and tau_twitch.__class__ is float
            and force_decay.__class__ is float
            and dt.__class__ is float
            and (
                drive
                + v
                + v_rest
                + v_reset
                + v_threshold
                + tau_m
                + adapt
                + tau_adapt
                + a_adapt
                + gain
                + force
                + twitch_amp
                + tau_twitch
                + force_decay
                + dt
            )
            * 0.0
            == 0.0
            and -150.0 <= v <= 100.0
            and -150.0 <= v_rest <= 100.0
            and -150.0 <= v_reset <= 100.0
            and -150.0 <= v_threshold <= 100.0
            and v_reset < v_threshold
            and 0.0 <= force <= 1.0
            and tau_m > 0.0
            and tau_adapt > 0.0
            and tau_twitch > 0.0
            and dt > 0.0
            and gain >= 0.0
            and twitch_amp >= 0.0
        ):
            drive = _require_finite_float(drive, "drive")
            self._validate_runtime()

        next_force = self.force * _exp(-self.dt / self.tau_twitch)
        input_drive = self.gain * max(0.0, drive) - self.adapt
        v_target = self.v_rest + input_drive
        v_candidate = _relax(self.v, v_target, self.tau_m, self.dt)
        if not _valid_voltage(v_candidate):
            raise ValueError(
                f"candidate membrane voltage must be finite and within [-150.0, 100.0], "
                f"got {v_candidate!r}."
            )

        adapt_target = self.a_adapt * (v_candidate - self.v_rest)
        adapt_candidate = _relax(self.adapt, adapt_target, self.tau_adapt, self.dt)
        if not _isfinite(adapt_candidate):
            raise ValueError(f"candidate adaptation state must be finite, got {adapt_candidate!r}.")

        spike = 0
        if v_candidate >= self.v_threshold:
            v_candidate = self.v_reset
            next_force = min(1.0, next_force + self.twitch_amp)
            spike = 1

        self.v = v_candidate
        self.adapt = adapt_candidate
        self.force = next_force
        return spike

    def reset(self) -> None:
        """Return ``v`` to ``v_rest`` and clear the adaptation and the force.

        Raises
        ------
        TypeError
            If ``v_rest`` is a boolean or not a real number.
        ValueError
            If ``v_rest`` is non-finite or outside ``[-150, 100]`` mV.
        """
        v_rest = _require_bounded_float(self.v_rest, "v_rest", -150.0, 100.0)
        self.v = v_rest
        self.adapt = 0.0
        self.force = 0.0
