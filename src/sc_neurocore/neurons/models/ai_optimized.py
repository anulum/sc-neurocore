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
from collections import deque
from dataclasses import dataclass, field


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

    def step(self, current: float) -> int:
        """Advance one Euler step of ``dt`` and report whether the fast compartment fired.

        The fast compartment integrates the current first; the spike test uses
        the threshold lowered by the slow compartment's value before this step.
        The medium compartment then integrates the spike and the slow one the
        medium compartment. A spike resets only the fast compartment to 0.

        Parameters
        ----------
        current : float
            Input current of this step.

        Returns
        -------
        int
            1 when ``v_fast >= theta_base - gamma * v_slow``, otherwise 0.
        """
        self.v_fast += (-self.v_fast + current) / self.tau_fast * self.dt
        theta_eff = self.theta_base - self.gamma * self.v_slow
        fired = int(self.v_fast >= theta_eff)
        self.v_medium += (-self.v_medium + self.alpha * fired) / self.tau_medium * self.dt
        self.v_slow += (-self.v_slow + self.beta * self.v_medium) / self.tau_slow * self.dt
        if fired:
            self.v_fast = 0.0
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

    def step(self, current: float) -> int:
        """Gate the current, integrate it for one Euler step and test the threshold.

        Parameters
        ----------
        current : float
            Input current of this step.

        Returns
        -------
        int
            1 when ``v`` reaches ``theta`` (``v`` is then reset to 0), otherwise 0.
        """
        gate = 1.0 / (1.0 + math.exp(-(self.w_key * current + self.w_query * self.v)))
        self.v += (-self.v + gate * current) / self.tau * self.dt
        if self.v >= self.theta:
            self.v = 0.0
            return 1
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

    def step(self, current: float) -> int:
        """Measure the surprise against the prediction, update both and test the threshold.

        The surprise is taken against the prediction before this step; the
        prediction then moves toward the current and ``v`` integrates the
        surprise.

        Parameters
        ----------
        current : float
            Input current of this step.

        Returns
        -------
        int
            1 when ``v`` reaches ``theta`` (``v`` is then reset to 0), otherwise 0.
        """
        surprise = abs(current - self.pred)
        self.pred += (current - self.pred) / self.tau_pred * self.dt
        self.v += (-self.v + surprise) / self.tau * self.dt
        if self.v >= self.theta:
            self.v = 0.0
            return 1
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

    def step(self, current: float) -> int:
        """Set the time constant from the recent rate, integrate one Euler step and test the threshold.

        Parameters
        ----------
        current : float
            Input current of this step.

        Returns
        -------
        int
            1 when ``v`` reaches ``theta`` (``v`` is then reset to 0), otherwise
            0; the result is appended to the history.
        """
        self._step_count += 1
        n_spikes = sum(self._history)
        rate = n_spikes / max(len(self._history), 1)
        tau_eff = self.tau * (1.0 + rate / self.target_rate)
        self.v += (-self.v + current) / tau_eff * self.dt
        if self.v >= self.theta:
            self.v = 0.0
            self._history.append(1)
            return 1
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

    def step(self, current: float) -> int:
        """Advance the phase, integrate the amplitude one Euler step and test the product.

        Parameters
        ----------
        current : float
            Input current of this step.

        Returns
        -------
        int
            1 when ``amplitude * cos(phi) > theta``, otherwise 0.
        """
        self.phi += self.omega * self.dt
        self.amplitude += (-self.amplitude + current) / self.tau * self.dt
        if self.amplitude * math.cos(self.phi) > self.theta:
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

    def step(self, current: float) -> int:
        """Decide the spike from ``v`` before this step's input, then update ``v``.

        Because the spike is tested first, a current that lifts ``v`` over
        ``theta`` shows as a spike on the following step.

        Parameters
        ----------
        current : float
            Input current of this step.

        Returns
        -------
        int
            1 when ``v >= theta`` at the start of the step, otherwise 0.
        """
        spike = int(self.v >= self.theta)
        self.v = self.alpha * self.v * (1 - spike) + current
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
        """
        return 1.0 / (1.0 + self.beta * abs(self.v - self.theta)) ** 2


@dataclass
class ContinuousAttractorNeuron:
    """Ring attractor for continuous working memory.

    u_i += (-u_i + f(sum_j w_ij u_j) + I_i) / tau * dt
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
        """Fill ``u`` with zeros and build the ring weights when they were not given."""
        if not self.u:
            self.u = [0.0] * self.n_units
        if not self._weights:
            self._build_weights()

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
        r = max(0.0, x)
        return r * r / (1.0 + r * r)

    def step(self, current: float) -> int:
        """Advance every unit one Euler step from the activities before the step.

        Parameters
        ----------
        current : float
            Input current added to every unit alike.

        Returns
        -------
        int
            1 when the largest unit activity exceeds 1.0 after the step, otherwise 0.
        """
        new_u = [0.0] * self.n_units
        for i in range(self.n_units):
            recurrent = sum(
                self._weights[i][j] * self._activation(self.u[j]) for j in range(self.n_units)
            )
            new_u[i] = self.u[i] + (-self.u[i] + recurrent + current) / self.tau * self.dt
        self.u = new_u
        peak = max(self.u)
        return int(peak > 1.0)

    def bump_position(self) -> int:
        """Return the index of the most active unit (the first one on a tie).

        Returns
        -------
        int
            Position of the activity bump on the ring, from 0 to ``n_units - 1``.
        """
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

    def step(self, current: float) -> int:
        """Integrate the current one Euler step and test the threshold.

        Parameters
        ----------
        current : float
            Input current of this step.

        Returns
        -------
        int
            1 when ``v`` reaches ``theta`` (``v`` is then reset to 0), otherwise 0.
        """
        self.v += (-self.v + current) / self.tau * self.dt
        if self.v >= self.theta:
            self.v = 0.0
            return 1
        return 0

    def update_meta(self, reward: float) -> None:
        """Update the error trace from one reward and move the expectation at the meta rate.

        The error is ``|reward - expected_reward|``; the meta rate is computed
        from the updated trace, then ``expected_reward`` moves toward the reward
        by that rate.

        Parameters
        ----------
        reward : float
            Reward observed for this update.
        """
        error = abs(reward - self.expected_reward)
        self.error_trace += (-self.error_trace + error) / self.tau_meta * self.dt
        meta_lr = self.lr0 / (1.0 + math.exp(-self.kappa * (self.error_trace - self.target_error)))
        self.expected_reward += meta_lr * (reward - self.expected_reward)

    @property
    def meta_lr(self) -> float:
        """Current meta-learning rate, ``lr0 * sigmoid(kappa * (error_trace - target_error))``.

        Returns
        -------
        float
            A rate between 0 and ``lr0``.
        """
        return self.lr0 / (1.0 + math.exp(-self.kappa * (self.error_trace - self.target_error)))

    def reset(self) -> None:
        """Set ``v``, the error trace and the expected reward to 0."""
        self.v = 0.0
        self.error_trace = 0.0
        self.expected_reward = 0.0
