# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AI-optimized spiking neuron models (original designs)

"""Eight novel neuron models designed for AI workloads, not biological simulation.

Each class is a small dataclass with ``step(current) -> int`` (1 for a spike,
0 otherwise) and ``reset()``. All eight are original SC-NeuroCore designs with
no external publication. The Rust engine has a counterpart of each with the
same step rule (``sc_neurocore_engine``) except ``SelfReferentialNeuron``: once
it fires within its first 50 steps, the two trains can stay different for the
rest of the run (see that class).
"""

from __future__ import annotations

import math
import numbers
from collections import deque
from dataclasses import dataclass, field


def _require_finite_float(value: object, name: str) -> float:
    """Validate that ``value`` is a finite real scalar (not ``bool``) and return ``float``."""
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite real number, not bool")
    if isinstance(value, numbers.Real):
        try:
            coerced = float(value)
        except OverflowError as exc:
            raise FloatingPointError(f"{name} overflowed finite floating-point range") from exc
        if not math.isfinite(coerced):
            raise ValueError(f"{name} must be finite")
        return coerced
    raise TypeError(f"{name} must be a finite real number")


def _require_int(value: object, name: str) -> int:
    """Validate that ``value`` is a non-boolean integer and return ``int``."""
    if type(value) is int:
        return value
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real) and not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    raise TypeError(f"{name} must be an integer")


def _safe_sigmoid(x: float) -> float:
    """Evaluate ``1 / (1 + exp(-x))`` without overflowing on large negative ``x``."""
    if x >= -700.0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


@dataclass
class MultiTimescaleNeuron:
    """Three-compartment memory neuron with fast/medium/slow timescales.

    dv_fast/dt   = (-v_fast + I) / tau_fast
    dv_medium/dt = (-v_medium + alpha * spike_fast) / tau_medium
    dv_slow/dt   = (-v_slow + beta * v_medium) / tau_slow
    theta_eff    = theta_base - gamma * v_slow

    Spike when v_fast >= theta_eff. The slow compartment accumulates
    context over seconds, modulating excitability.

    Reference: Custom SC-NeuroCore model — no external publication.
    """

    v_fast: float = 0.0
    v_medium: float = 0.0
    v_slow: float = 0.0
    tau_fast: float = 5.0
    tau_medium: float = 200.0
    tau_slow: float = 10000.0
    alpha: float = 10.0
    beta: float = 0.05
    gamma: float = 0.3
    theta_base: float = 1.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        """Validate all compartments, time constants and positive timestep."""
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        self.v_fast = _require_finite_float(self.v_fast, "v_fast")
        self.v_medium = _require_finite_float(self.v_medium, "v_medium")
        self.v_slow = _require_finite_float(self.v_slow, "v_slow")
        self.tau_fast = _require_finite_float(self.tau_fast, "tau_fast")
        self.tau_medium = _require_finite_float(self.tau_medium, "tau_medium")
        self.tau_slow = _require_finite_float(self.tau_slow, "tau_slow")
        self.alpha = _require_finite_float(self.alpha, "alpha")
        self.beta = _require_finite_float(self.beta, "beta")
        self.gamma = _require_finite_float(self.gamma, "gamma")
        self.theta_base = _require_finite_float(self.theta_base, "theta_base")
        self.dt = _require_finite_float(self.dt, "dt")
        for name in ("tau_fast", "tau_medium", "tau_slow", "dt"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def step(self, current: float) -> int:
        """Advance one Euler step of ``dt`` and report whether the fast compartment fired.

        The fast compartment integrates the current first; the spike test uses
        the threshold lowered by the slow compartment's value before this step.
        The medium compartment then integrates the spike and the slow one the
        medium compartment. A spike resets only the fast compartment to 0.

        Parameters
        ----------
        current : float
            Input current of this step; must be finite.

        Returns
        -------
        int
            1 when ``v_fast >= theta_base - gamma * v_slow``, otherwise 0.

        Raises
        ------
        ValueError
            If ``current`` or any state/parameter is non-finite, or any time
            constant or ``dt`` is not positive. State is left unchanged.
        TypeError
            If ``current`` or any state/parameter is a boolean or non-real value.
            State is left unchanged.
        FloatingPointError
            If a candidate compartment state overflows finite range.
            State is left unchanged.
        """
        v_fast = self.v_fast
        v_medium = self.v_medium
        v_slow = self.v_slow
        tau_fast = self.tau_fast
        tau_medium = self.tau_medium
        tau_slow = self.tau_slow
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        theta_base = self.theta_base
        dt = self.dt
        if not (
            current.__class__ is float
            and math.isfinite(current)
            and v_fast.__class__ is float
            and v_medium.__class__ is float
            and v_slow.__class__ is float
            and tau_fast.__class__ is float
            and tau_medium.__class__ is float
            and tau_slow.__class__ is float
            and alpha.__class__ is float
            and beta.__class__ is float
            and gamma.__class__ is float
            and theta_base.__class__ is float
            and dt.__class__ is float
            and math.isfinite(v_fast)
            and math.isfinite(v_medium)
            and math.isfinite(v_slow)
            and math.isfinite(tau_fast)
            and math.isfinite(tau_medium)
            and math.isfinite(tau_slow)
            and math.isfinite(alpha)
            and math.isfinite(beta)
            and math.isfinite(gamma)
            and math.isfinite(theta_base)
            and math.isfinite(dt)
            and tau_fast > 0.0
            and tau_medium > 0.0
            and tau_slow > 0.0
            and dt > 0.0
        ):
            current = _require_finite_float(current, "current")
            self._validate_configuration()
            v_fast = self.v_fast
            v_medium = self.v_medium
            v_slow = self.v_slow
            tau_fast = self.tau_fast
            tau_medium = self.tau_medium
            tau_slow = self.tau_slow
            alpha = self.alpha
            beta = self.beta
            gamma = self.gamma
            theta_base = self.theta_base
            dt = self.dt

        next_v_fast = v_fast + (-v_fast + current) / tau_fast * dt
        theta_eff = theta_base - gamma * v_slow
        fired = int(next_v_fast >= theta_eff)
        next_v_medium = v_medium + (-v_medium + alpha * fired) / tau_medium * dt
        next_v_slow = v_slow + (-v_slow + beta * next_v_medium) / tau_slow * dt
        if not (
            math.isfinite(next_v_fast)
            and math.isfinite(theta_eff)
            and math.isfinite(next_v_medium)
            and math.isfinite(next_v_slow)
        ):
            raise FloatingPointError("MultiTimescaleNeuron state overflowed finite range")
        self.v_fast = 0.0 if fired else next_v_fast
        self.v_medium = next_v_medium
        self.v_slow = next_v_slow
        return fired

    def reset(self) -> None:
        """Set the three compartments to 0; the parameters are kept."""
        self.v_fast = 0.0
        self.v_medium = 0.0
        self.v_slow = 0.0


@dataclass
class AttentionGatedNeuron:
    """Spiking neuron whose input passes a sigmoid attention gate.

    gate = sigmoid(w_key * I + w_query * v)
    dv/dt = (-v + gate * I) / tau

    The key and query weights decide which input magnitudes pass and which
    are suppressed. They are parameters: this class does not change them,
    so any learning of them is the caller's.
    """

    v: float = 0.0
    w_key: float = 1.0
    w_query: float = 0.5
    tau: float = 10.0
    theta: float = 1.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        """Validate voltage, attention weights, positive ``tau`` and ``dt``."""
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        self.v = _require_finite_float(self.v, "v")
        self.w_key = _require_finite_float(self.w_key, "w_key")
        self.w_query = _require_finite_float(self.w_query, "w_query")
        self.tau = _require_finite_float(self.tau, "tau")
        self.theta = _require_finite_float(self.theta, "theta")
        self.dt = _require_finite_float(self.dt, "dt")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")

    def step(self, current: float) -> int:
        """Gate the current, integrate it for one Euler step and test the threshold.

        Parameters
        ----------
        current : float
            Input current of this step; must be finite.

        Returns
        -------
        int
            1 when ``v`` reaches ``theta`` (``v`` is then reset to 0), otherwise 0.

        Raises
        ------
        ValueError
            If ``current`` or any state/parameter is non-finite, or ``tau`` or
            ``dt`` is not positive. State is left unchanged.
        TypeError
            If ``current`` or any state/parameter is a boolean or non-real value.
            State is left unchanged.
        FloatingPointError
            If the candidate membrane potential overflows finite range.
            State is left unchanged.
        """
        v = self.v
        w_key = self.w_key
        w_query = self.w_query
        tau = self.tau
        theta = self.theta
        dt = self.dt
        if not (
            current.__class__ is float
            and math.isfinite(current)
            and v.__class__ is float
            and w_key.__class__ is float
            and w_query.__class__ is float
            and tau.__class__ is float
            and theta.__class__ is float
            and dt.__class__ is float
            and math.isfinite(v)
            and math.isfinite(w_key)
            and math.isfinite(w_query)
            and math.isfinite(tau)
            and math.isfinite(theta)
            and math.isfinite(dt)
            and tau > 0.0
            and dt > 0.0
        ):
            current = _require_finite_float(current, "current")
            self._validate_configuration()
            v = self.v
            w_key = self.w_key
            w_query = self.w_query
            tau = self.tau
            theta = self.theta
            dt = self.dt

        gate = _safe_sigmoid(w_key * current + w_query * v)
        next_v = v + (-v + gate * current) / tau * dt
        if not math.isfinite(next_v):
            raise FloatingPointError("AttentionGatedNeuron state overflowed finite range")
        if next_v >= theta:
            self.v = 0.0
            return 1
        self.v = next_v
        return 0

    def reset(self) -> None:
        """Set ``v`` to 0; the weights and other parameters are kept."""
        self.v = 0.0


@dataclass
class PredictiveCodingNeuron:
    """Fires only on prediction errors.

    dpred/dt = (I - pred) / tau_pred
    surprise = |I - pred|
    dv/dt    = (-v + surprise) / tau

    Silent when input matches prediction. Fires on novel stimuli whose
    surprise, integrated, reaches ``theta``: an input that differs from the
    prediction by less than ``theta`` does not fire while ``dt <= tau``, as with
    the defaults (a larger step overshoots).
    """

    v: float = 0.0
    pred: float = 0.0
    tau: float = 10.0
    tau_pred: float = 50.0
    theta: float = 1.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        """Validate state, threshold and positive ``tau``, ``tau_pred`` and ``dt``."""
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        self.v = _require_finite_float(self.v, "v")
        self.pred = _require_finite_float(self.pred, "pred")
        self.tau = _require_finite_float(self.tau, "tau")
        self.tau_pred = _require_finite_float(self.tau_pred, "tau_pred")
        self.theta = _require_finite_float(self.theta, "theta")
        self.dt = _require_finite_float(self.dt, "dt")
        for name in ("tau", "tau_pred", "dt"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def step(self, current: float) -> int:
        """Measure the surprise against the prediction, update both and test the threshold.

        The surprise is taken against the prediction before this step; the
        prediction then moves toward the current and ``v`` integrates the
        surprise.

        Parameters
        ----------
        current : float
            Input current of this step; must be finite.

        Returns
        -------
        int
            1 when ``v`` reaches ``theta`` (``v`` is then reset to 0), otherwise 0.

        Raises
        ------
        ValueError
            If ``current`` or any state/parameter is non-finite, or ``tau``,
            ``tau_pred`` or ``dt`` is not positive. State is left unchanged.
        TypeError
            If ``current`` or any state/parameter is a boolean or non-real value.
            State is left unchanged.
        FloatingPointError
            If a candidate state overflows finite range. State is left unchanged.
        """
        v = self.v
        pred = self.pred
        tau = self.tau
        tau_pred = self.tau_pred
        theta = self.theta
        dt = self.dt
        if not (
            current.__class__ is float
            and math.isfinite(current)
            and v.__class__ is float
            and pred.__class__ is float
            and tau.__class__ is float
            and tau_pred.__class__ is float
            and theta.__class__ is float
            and dt.__class__ is float
            and math.isfinite(v)
            and math.isfinite(pred)
            and math.isfinite(tau)
            and math.isfinite(tau_pred)
            and math.isfinite(theta)
            and math.isfinite(dt)
            and tau > 0.0
            and tau_pred > 0.0
            and dt > 0.0
        ):
            current = _require_finite_float(current, "current")
            self._validate_configuration()
            v = self.v
            pred = self.pred
            tau = self.tau
            tau_pred = self.tau_pred
            theta = self.theta
            dt = self.dt

        surprise = abs(current - pred)
        next_pred = pred + (current - pred) / tau_pred * dt
        next_v = v + (-v + surprise) / tau * dt
        if not (math.isfinite(surprise) and math.isfinite(next_pred) and math.isfinite(next_v)):
            raise FloatingPointError("PredictiveCodingNeuron state overflowed finite range")
        self.pred = next_pred
        if next_v >= theta:
            self.v = 0.0
            return 1
        self.v = next_v
        return 0

    def reset(self) -> None:
        """Set ``v`` and the prediction to 0."""
        self.v = 0.0
        self.pred = 0.0


@dataclass
class SelfReferentialNeuron:
    """Leaky integrate-and-fire neuron whose time constant follows its own firing rate.

    rate    = spikes among the last steps / number of those steps (at most 50)
    tau_eff = tau * (1 + rate / target_rate)
    dv/dt   = (-v + I) / tau_eff

    The more it has fired recently, the slower it integrates, which damps
    its own rate. The threshold ``theta`` is fixed. The history holds the
    last 50 steps whatever ``window`` says: the field is stored but not
    used. The Rust engine's counterpart divides by the full 50-step window
    from the first step, so the two part when the neuron fires within its
    first 50 steps and can stay apart afterwards: under a constant current
    of 2.0 they first differ at the 15th step and differ at 155 of the first
    1,000 steps.
    """

    v: float = 0.0
    tau: float = 10.0
    theta: float = 1.0
    target_rate: float = 0.1
    window: int = 50
    dt: float = 1.0
    _history: deque[int] = field(default_factory=lambda: deque(maxlen=50))
    _step_count: int = 0

    def __post_init__(self) -> None:
        """Validate state, positive ``tau``, ``target_rate``, ``dt`` and integer counters."""
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        self.v = _require_finite_float(self.v, "v")
        self.tau = _require_finite_float(self.tau, "tau")
        self.theta = _require_finite_float(self.theta, "theta")
        self.target_rate = _require_finite_float(self.target_rate, "target_rate")
        self.dt = _require_finite_float(self.dt, "dt")
        self.window = _require_int(self.window, "window")
        self._step_count = _require_int(self._step_count, "_step_count")
        for name in ("tau", "target_rate", "dt"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.window <= 0:
            raise ValueError("window must be positive")
        if self._step_count < 0:
            raise ValueError("_step_count must be non-negative")

    def step(self, current: float) -> int:
        """Set the time constant from the recent rate, integrate one Euler step and test the threshold.

        Parameters
        ----------
        current : float
            Input current of this step; must be finite.

        Returns
        -------
        int
            1 when ``v`` reaches ``theta`` (``v`` is then reset to 0), otherwise
            0; the result is appended to the history.

        Raises
        ------
        ValueError
            If ``current`` or any state/parameter is non-finite, or ``tau``,
            ``target_rate``, ``dt`` or ``window`` is not positive, or
            ``_step_count`` is negative. State and history are left unchanged.
        TypeError
            If ``current`` or any field has an invalid type. State and history
            are left unchanged.
        FloatingPointError
            If the candidate state or effective time constant overflows finite
            range. State and history are left unchanged.
        """
        v = self.v
        tau = self.tau
        theta = self.theta
        target_rate = self.target_rate
        dt = self.dt
        window = self.window
        step_count = self._step_count
        if not (
            current.__class__ is float
            and math.isfinite(current)
            and v.__class__ is float
            and tau.__class__ is float
            and theta.__class__ is float
            and target_rate.__class__ is float
            and dt.__class__ is float
            and window.__class__ is int
            and step_count.__class__ is int
            and math.isfinite(v)
            and math.isfinite(tau)
            and math.isfinite(theta)
            and math.isfinite(target_rate)
            and math.isfinite(dt)
            and tau > 0.0
            and target_rate > 0.0
            and dt > 0.0
            and window > 0
            and step_count >= 0
        ):
            current = _require_finite_float(current, "current")
            self._validate_configuration()
            v = self.v
            tau = self.tau
            theta = self.theta
            target_rate = self.target_rate
            dt = self.dt
            step_count = self._step_count

        n_spikes = sum(self._history)
        rate = n_spikes / max(len(self._history), 1)
        tau_eff = tau * (1.0 + rate / target_rate)
        next_v = v + (-v + current) / tau_eff * dt
        if not (math.isfinite(tau_eff) and math.isfinite(next_v)):
            raise FloatingPointError("SelfReferentialNeuron state overflowed finite range")

        self._step_count = step_count + 1
        if next_v >= theta:
            self.v = 0.0
            self._history.append(1)
            return 1
        self.v = next_v
        self._history.append(0)
        return 0

    def reset(self) -> None:
        """Set ``v`` to 0 and clear the spike history and the step count."""
        self.v = 0.0
        self._history.clear()
        self._step_count = 0


@dataclass
class CompositionalBindingNeuron:
    """Phase-coding neuron meant for variable binding.

    dphi/dt = omega
    dA/dt   = (-A + I) / tau
    Spike when A * cos(phi) > theta.

    The idea is that two neurons in phase encode bound concepts and a phase
    offset encodes relational structure. This class runs one neuron with a
    freely advancing phase: it takes no input phase, ``coupling`` is stored
    but not used, and nothing resets on a spike, so it fires on every step
    while ``A * cos(phi)`` stays above ``theta``.
    """

    phi: float = 0.0
    amplitude: float = 0.0
    omega: float = 0.1
    coupling: float = 0.5
    tau: float = 10.0
    theta: float = 0.8
    dt: float = 1.0

    def __post_init__(self) -> None:
        """Validate phase, amplitude, parameters and positive ``tau`` and ``dt``."""
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        self.phi = _require_finite_float(self.phi, "phi")
        self.amplitude = _require_finite_float(self.amplitude, "amplitude")
        self.omega = _require_finite_float(self.omega, "omega")
        self.coupling = _require_finite_float(self.coupling, "coupling")
        self.tau = _require_finite_float(self.tau, "tau")
        self.theta = _require_finite_float(self.theta, "theta")
        self.dt = _require_finite_float(self.dt, "dt")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")

    def step(self, current: float) -> int:
        """Advance the phase, integrate the amplitude one Euler step and test the product.

        Parameters
        ----------
        current : float
            Input current of this step; must be finite.

        Returns
        -------
        int
            1 when ``amplitude * cos(phi) > theta``, otherwise 0.

        Raises
        ------
        ValueError
            If ``current`` or any state/parameter is non-finite, or ``tau`` or
            ``dt`` is not positive. State is left unchanged.
        TypeError
            If ``current`` or any state/parameter is a boolean or non-real value.
            State is left unchanged.
        FloatingPointError
            If candidate phase or amplitude overflows finite range.
            State is left unchanged.
        """
        phi = self.phi
        amplitude = self.amplitude
        omega = self.omega
        coupling = self.coupling
        tau = self.tau
        theta = self.theta
        dt = self.dt
        if not (
            current.__class__ is float
            and math.isfinite(current)
            and phi.__class__ is float
            and amplitude.__class__ is float
            and omega.__class__ is float
            and coupling.__class__ is float
            and tau.__class__ is float
            and theta.__class__ is float
            and dt.__class__ is float
            and math.isfinite(phi)
            and math.isfinite(amplitude)
            and math.isfinite(omega)
            and math.isfinite(coupling)
            and math.isfinite(tau)
            and math.isfinite(theta)
            and math.isfinite(dt)
            and tau > 0.0
            and dt > 0.0
        ):
            current = _require_finite_float(current, "current")
            self._validate_configuration()
            phi = self.phi
            amplitude = self.amplitude
            omega = self.omega
            tau = self.tau
            theta = self.theta
            dt = self.dt

        next_phi = phi + omega * dt
        next_amplitude = amplitude + (-amplitude + current) / tau * dt
        if not (math.isfinite(next_phi) and math.isfinite(next_amplitude)):
            raise FloatingPointError("CompositionalBindingNeuron state overflowed finite range")
        projection = next_amplitude * math.cos(next_phi)
        self.phi = next_phi
        self.amplitude = next_amplitude
        if projection > theta:
            return 1
        return 0

    def reset(self) -> None:
        """Set the phase and the amplitude to 0."""
        self.phi = 0.0
        self.amplitude = 0.0


@dataclass
class DifferentiableSurrogateNeuron:
    """Spiking neuron with a surrogate gradient for gradient-based training.

    Forward: spike = int(v >= theta)
    Backward: surrogate = 1 / (1 + beta * |v - theta|)^2   [conceptual]
    v = alpha * v * (1 - spike) + I

    alpha (decay), beta (steepness) and theta (threshold) are the parameters
    a training procedure would adjust. This class does not adjust them: it
    computes the forward step and, through ``surrogate_grad``, the surrogate
    derivative at the current ``v``.
    """

    v: float = 0.0
    alpha: float = 0.9
    beta: float = 5.0
    theta: float = 1.0

    def __post_init__(self) -> None:
        """Validate state, threshold, decay and non-negative ``beta``."""
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        self.v = _require_finite_float(self.v, "v")
        self.alpha = _require_finite_float(self.alpha, "alpha")
        self.beta = _require_finite_float(self.beta, "beta")
        self.theta = _require_finite_float(self.theta, "theta")
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative")

    def step(self, current: float) -> int:
        """Decide the spike from ``v`` before this step's input, then update ``v``.

        Because the spike is tested first, a current that lifts ``v`` over
        ``theta`` shows as a spike on the following step.

        Parameters
        ----------
        current : float
            Input current of this step; must be finite.

        Returns
        -------
        int
            1 when ``v >= theta`` at the start of the step, otherwise 0.

        Raises
        ------
        ValueError
            If ``current`` or any state/parameter is non-finite, or ``beta`` is
            negative. State is left unchanged.
        TypeError
            If ``current`` or any state/parameter is a boolean or non-real value.
            State is left unchanged.
        FloatingPointError
            If the candidate membrane potential overflows finite range.
            State is left unchanged.
        """
        v = self.v
        alpha = self.alpha
        beta = self.beta
        theta = self.theta
        if not (
            current.__class__ is float
            and math.isfinite(current)
            and v.__class__ is float
            and alpha.__class__ is float
            and beta.__class__ is float
            and theta.__class__ is float
            and math.isfinite(v)
            and math.isfinite(alpha)
            and math.isfinite(beta)
            and math.isfinite(theta)
            and beta >= 0.0
        ):
            current = _require_finite_float(current, "current")
            self._validate_configuration()
            v = self.v
            alpha = self.alpha
            theta = self.theta

        spike = int(v >= theta)
        next_v = alpha * v * (1 - spike) + current
        if not math.isfinite(next_v):
            raise FloatingPointError("DifferentiableSurrogateNeuron state overflowed finite range")
        self.v = next_v
        return spike

    def reset(self) -> None:
        """Set ``v`` to 0; the parameters are kept."""
        self.v = 0.0

    def surrogate_grad(self) -> float:
        """Smooth surrogate gradient for backprop.

        Returns
        -------
        float
            ``1 / (1 + beta * |v - theta|) ** 2`` at the current ``v``: 1 at the
            threshold, falling with the distance from it.

        Raises
        ------
        ValueError
            If ``v``, ``beta`` or ``theta`` is non-finite, or ``beta`` is negative.
        TypeError
            If ``v``, ``beta`` or ``theta`` is a boolean or non-real value.
        """
        self._validate_configuration()
        denom = 1.0 + self.beta * abs(self.v - self.theta)
        if not math.isfinite(denom) or denom > 1e154:
            return 0.0
        return 1.0 / (denom * denom)


@dataclass
class ContinuousAttractorNeuron:
    """Ring attractor for continuous working memory.

    u_i += (-u_i + sum_j w_ij f(u_j) + I) / tau * dt
    f(x) = max(0,x)^2 / (1 + max(0,x)^2)
    w_ij = A * exp(-d_ij^2 / (2*sigma_e^2)) - B   (Mexican hat)

    ``A`` is ``excitation``, ``B`` is ``inhibition`` and ``d_ij`` the distance
    around the ring. An activity bump given in the initial ``u`` persists at
    its position. The scalar current of ``step`` drives every unit alike, so
    it cannot place or move a bump. ``step`` reports whether the peak exceeds
    1.0; ``bump_position`` gives the position of the activity bump.
    """

    n_units: int = 16
    tau: float = 10.0
    sigma_e: float = 1.0
    excitation: float = 4.0
    inhibition: float = 0.5
    dt: float = 1.0
    u: list[float] = field(default_factory=list)
    _weights: list[list[float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate ring parameters, fill ``u`` with zeros and build weights if absent."""
        self._validate_parameters()
        if not self.u:
            self.u = [0.0] * self.n_units
        elif len(self.u) != self.n_units:
            raise ValueError(f"u must have length {self.n_units}, got {len(self.u)}")
        if not self._weights:
            self._build_weights()
        else:
            self._validate_weight_shape()

    def _validate_parameters(self) -> None:
        self.n_units = _require_int(self.n_units, "n_units")
        if self.n_units <= 0:
            raise ValueError("n_units must be positive")
        self.tau = _require_finite_float(self.tau, "tau")
        self.sigma_e = _require_finite_float(self.sigma_e, "sigma_e")
        self.excitation = _require_finite_float(self.excitation, "excitation")
        self.inhibition = _require_finite_float(self.inhibition, "inhibition")
        self.dt = _require_finite_float(self.dt, "dt")
        for name in ("tau", "sigma_e", "dt"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")

    def _validate_weight_shape(self) -> None:
        if len(self._weights) != self.n_units:
            raise ValueError(f"_weights must have {self.n_units} rows, got {len(self._weights)}")
        for idx, row in enumerate(self._weights):
            if len(row) != self.n_units:
                raise ValueError(f"_weights[{idx}] must have length {self.n_units}, got {len(row)}")

    def _build_weights(self) -> None:
        """Build the Mexican-hat weights over circular distances on the ring."""
        n = self.n_units
        self._weights = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                d = min(abs(i - j), n - abs(i - j))
                self._weights[i][j] = (
                    self.excitation * math.exp(-d * d / (2.0 * self.sigma_e**2)) - self.inhibition
                )

    @staticmethod
    def _activation(x: float) -> float:
        """Return the saturating rate ``max(0, x)^2 / (1 + max(0, x)^2)``."""
        if x <= 0.0:
            return 0.0
        if x >= 1e154:
            return 1.0
        r_sq = x * x
        return r_sq / (1.0 + r_sq)

    def step(self, current: float) -> int:
        """Advance every unit one Euler step from the activities before the step.

        Each unit's recurrent input is summed left to right over the ring, as the engine sums
        it. The built-in ``sum`` compensates rounding from Python 3.12 on, which would make the
        last bits, and on an even ring the unit that holds the bump, depend on the Python version.

        Parameters
        ----------
        current : float
            Input current added to every unit alike; must be finite.

        Returns
        -------
        int
            1 when the largest unit activity exceeds 1.0 after the step, otherwise 0.

        Raises
        ------
        ValueError
            If ``current``, any ring parameter, any activity in ``u`` or any
            weight is non-finite, or ring dimensions/shapes are invalid.
            State is left unchanged.
        TypeError
            If ``current`` or any scalar field has an invalid type.
            State is left unchanged.
        FloatingPointError
            If any candidate unit activity overflows finite range.
            State is left unchanged.
        """
        drive = _require_finite_float(current, "current")
        self._validate_parameters()
        n = self.n_units
        if len(self.u) != n:
            raise ValueError(f"u must have length {n}, got {len(self.u)}")
        self._validate_weight_shape()
        u = self.u
        weights = self._weights
        rates = [0.0] * n
        for j in range(n):
            uj = u[j]
            if not (uj.__class__ is float and math.isfinite(uj)):
                uj = _require_finite_float(uj, f"u[{j}]")
            rates[j] = self._activation(uj)

        tau = self.tau
        dt = self.dt
        new_u = [0.0] * n
        for i in range(n):
            row = weights[i]
            recurrent = 0.0
            for j in range(n):
                w_ij = row[j]
                if not (w_ij.__class__ is float and math.isfinite(w_ij)):
                    w_ij = _require_finite_float(w_ij, f"_weights[{i}][{j}]")
                recurrent += w_ij * rates[j]
            cand = u[i] + (-u[i] + recurrent + drive) / tau * dt
            if not math.isfinite(cand):
                raise FloatingPointError("ContinuousAttractorNeuron state overflowed finite range")
            new_u[i] = cand
        self.u = new_u
        peak = max(self.u)
        return int(peak > 1.0)

    def bump_position(self) -> int:
        """Return the index of the most active unit, the first of equal maxima, as the engine does.

        Returns
        -------
        int
            Position of the activity bump on the ring, from 0 to ``n_units - 1``.

        Raises
        ------
        ValueError
            If ``u`` is empty.
        """
        if not self.u:
            raise ValueError("u must not be empty")
        return self.u.index(max(self.u))

    def reset(self) -> None:
        """Set every unit's activity to 0; the weights are kept."""
        self.u = [0.0] * self.n_units


@dataclass
class MetaPlasticNeuron:
    """Neuron with self-regulating meta-learning rate.

    dv/dt = (-v + I) / tau
    error_trace += (-error_trace + |reward - expected|) / tau_meta * dt
    meta_lr = lr0 * sigmoid(kappa * (error_trace - target_error))

    High error: faster learning. Low error: stabilize. The spiking part
    (``step``) is a leaky integrate-and-fire neuron that the meta state does
    not touch; the meta state moves only through ``update_meta``.
    """

    v: float = 0.0
    error_trace: float = 0.0
    expected_reward: float = 0.0
    tau: float = 10.0
    tau_meta: float = 500.0
    theta: float = 1.0
    lr0: float = 0.01
    kappa: float = 5.0
    target_error: float = 0.3
    dt: float = 1.0

    def __post_init__(self) -> None:
        """Validate state, positive time constants/``dt`` and non-negative ``lr0``."""
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        self.v = _require_finite_float(self.v, "v")
        self.error_trace = _require_finite_float(self.error_trace, "error_trace")
        self.expected_reward = _require_finite_float(self.expected_reward, "expected_reward")
        self.tau = _require_finite_float(self.tau, "tau")
        self.tau_meta = _require_finite_float(self.tau_meta, "tau_meta")
        self.theta = _require_finite_float(self.theta, "theta")
        self.lr0 = _require_finite_float(self.lr0, "lr0")
        self.kappa = _require_finite_float(self.kappa, "kappa")
        self.target_error = _require_finite_float(self.target_error, "target_error")
        self.dt = _require_finite_float(self.dt, "dt")
        for name in ("tau", "tau_meta", "dt"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.lr0 < 0.0:
            raise ValueError("lr0 must be non-negative")

    def step(self, current: float) -> int:
        """Integrate the current one Euler step and test the threshold.

        Parameters
        ----------
        current : float
            Input current of this step; must be finite.

        Returns
        -------
        int
            1 when ``v`` reaches ``theta`` (``v`` is then reset to 0), otherwise 0.

        Raises
        ------
        ValueError
            If ``current`` or any state/parameter is non-finite, or ``tau``,
            ``tau_meta`` or ``dt`` is not positive, or ``lr0`` is negative.
            State is left unchanged.
        TypeError
            If ``current`` or any state/parameter is a boolean or non-real value.
            State is left unchanged.
        FloatingPointError
            If the candidate membrane potential overflows finite range.
            State is left unchanged.
        """
        v = self.v
        tau = self.tau
        tau_meta = self.tau_meta
        theta = self.theta
        lr0 = self.lr0
        dt = self.dt
        if not (
            current.__class__ is float
            and math.isfinite(current)
            and v.__class__ is float
            and tau.__class__ is float
            and tau_meta.__class__ is float
            and theta.__class__ is float
            and lr0.__class__ is float
            and dt.__class__ is float
            and math.isfinite(v)
            and math.isfinite(tau)
            and math.isfinite(tau_meta)
            and math.isfinite(theta)
            and math.isfinite(lr0)
            and math.isfinite(dt)
            and tau > 0.0
            and tau_meta > 0.0
            and dt > 0.0
            and lr0 >= 0.0
        ):
            current = _require_finite_float(current, "current")
            self._validate_configuration()
            v = self.v
            tau = self.tau
            theta = self.theta
            dt = self.dt

        next_v = v + (-v + current) / tau * dt
        if not math.isfinite(next_v):
            raise FloatingPointError("MetaPlasticNeuron state overflowed finite range")
        if next_v >= theta:
            self.v = 0.0
            return 1
        self.v = next_v
        return 0

    def update_meta(self, reward: float) -> None:
        """Update the error trace from one reward and move the expectation at the meta rate.

        The error is ``|reward - expected_reward|``; the meta rate is computed
        from the updated trace, then ``expected_reward`` moves toward the reward
        by that rate.

        Parameters
        ----------
        reward : float
            Reward observed for this update; must be finite.

        Raises
        ------
        ValueError
            If ``reward`` or any state/parameter is non-finite, or time
            constants/``dt`` are not positive, or ``lr0`` is negative.
            Meta state is left unchanged.
        TypeError
            If ``reward`` or any state/parameter is a boolean or non-real value.
            Meta state is left unchanged.
        FloatingPointError
            If candidate error trace or expected reward overflows finite range.
            Meta state is left unchanged.
        """
        reward_val = _require_finite_float(reward, "reward")
        self._validate_configuration()
        error = abs(reward_val - self.expected_reward)
        next_error_trace = self.error_trace + (-self.error_trace + error) / self.tau_meta * self.dt
        meta_lr = self.lr0 * _safe_sigmoid(self.kappa * (next_error_trace - self.target_error))
        next_expected = self.expected_reward + meta_lr * (reward_val - self.expected_reward)
        if not (
            math.isfinite(error)
            and math.isfinite(next_error_trace)
            and math.isfinite(next_expected)
        ):
            raise FloatingPointError("MetaPlasticNeuron meta state overflowed finite range")
        self.error_trace = next_error_trace
        self.expected_reward = next_expected

    @property
    def meta_lr(self) -> float:
        """Current meta-learning rate, ``lr0 * sigmoid(kappa * (error_trace - target_error))``.

        Returns
        -------
        float
            A rate between 0 and ``lr0``.

        Raises
        ------
        ValueError
            If configuration or meta state is invalid or non-finite.
        TypeError
            If any numeric field is a boolean or non-real value.
        """
        self._validate_configuration()
        return self.lr0 * _safe_sigmoid(self.kappa * (self.error_trace - self.target_error))

    def reset(self) -> None:
        """Set ``v``, the error trace and the expected reward to 0."""
        self.v = 0.0
        self.error_trace = 0.0
        self.expected_reward = 0.0
