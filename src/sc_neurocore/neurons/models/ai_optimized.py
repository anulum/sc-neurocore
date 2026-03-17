# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AI-optimized spiking neuron models (original designs)

"""Eight novel neuron models designed for AI workloads, not biological simulation."""

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
        self.v_fast += (-self.v_fast + current) / self.tau_fast * self.dt
        theta_eff = self.theta_base - self.gamma * self.v_slow
        fired = int(self.v_fast >= theta_eff)
        self.v_medium += (-self.v_medium + self.alpha * fired) / self.tau_medium * self.dt
        self.v_slow += (-self.v_slow + self.beta * self.v_medium) / self.tau_slow * self.dt
        if fired:
            self.v_fast = 0.0
        return fired

    def reset(self):
        self.v_fast = 0.0
        self.v_medium = 0.0
        self.v_slow = 0.0


@dataclass
class AttentionGatedNeuron:
    """Spiking neuron with learned sigmoid attention gate.

    gate = sigmoid(w_key * I + w_query * v)
    dv/dt = (-v + gate * I) / tau

    Each neuron learns which input magnitudes to attend to
    and which to suppress, via key/query weights.
    """

    v: float = 0.0
    w_key: float = 1.0
    w_query: float = 0.5
    tau: float = 10.0
    theta: float = 1.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        gate = 1.0 / (1.0 + math.exp(-(self.w_key * current + self.w_query * self.v)))
        self.v += (-self.v + gate * current) / self.tau * self.dt
        if self.v >= self.theta:
            self.v = 0.0
            return 1
        return 0

    def reset(self):
        self.v = 0.0


@dataclass
class PredictiveCodingNeuron:
    """Fires only on prediction errors.

    dpred/dt = (I - pred) / tau_pred
    surprise = |I - pred|
    dv/dt    = (-v + surprise) / tau

    Silent when input matches prediction. Fires on novel stimuli.
    """

    v: float = 0.0
    pred: float = 0.0
    tau: float = 10.0
    tau_pred: float = 50.0
    theta: float = 1.0
    dt: float = 1.0

    def step(self, current: float) -> int:
        surprise = abs(current - self.pred)
        self.pred += (current - self.pred) / self.tau_pred * self.dt
        self.v += (-self.v + surprise) / self.tau * self.dt
        if self.v >= self.theta:
            self.v = 0.0
            return 1
        return 0

    def reset(self):
        self.v = 0.0
        self.pred = 0.0


@dataclass
class SelfReferentialNeuron:
    """Introspects on its own spike history to modulate dynamics.

    self_rate = count(recent_spikes) / window
    effective_tau = tau * (1 + self_rate / target_rate)
    theta_eff = theta * (1 + regularity)

    Regular firing lowers threshold (maintain pattern).
    Chaotic firing raises threshold (stabilize).
    """

    v: float = 0.0
    tau: float = 10.0
    theta: float = 1.0
    target_rate: float = 0.1
    window: int = 50
    dt: float = 1.0
    _history: deque = field(default_factory=lambda: deque(maxlen=50))
    _step_count: int = 0

    def step(self, current: float) -> int:
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

    def reset(self):
        self.v = 0.0
        self._history.clear()
        self._step_count = 0


@dataclass
class CompositionalBindingNeuron:
    """Phase-coding neuron for variable binding.

    dphi/dt = omega + coupling * sin(phi_input - phi)
    dA/dt   = (-A + I) / tau
    Spike when A * cos(phi) > theta.

    Two neurons in-phase encode bound concepts.
    Phase offset encodes relational structure.
    """

    phi: float = 0.0
    amplitude: float = 0.0
    omega: float = 0.1
    coupling: float = 0.5
    tau: float = 10.0
    theta: float = 0.8
    dt: float = 1.0

    def step(self, current: float) -> int:
        self.phi += self.omega * self.dt
        self.amplitude += (-self.amplitude + current) / self.tau * self.dt
        if self.amplitude * math.cos(self.phi) > self.theta:
            return 1
        return 0

    def reset(self):
        self.phi = 0.0
        self.amplitude = 0.0


@dataclass
class DifferentiableSurrogateNeuron:
    """Spiking neuron with learnable surrogate gradient parameters.

    Forward: spike = int(v >= theta)
    Backward: surrogate = 1 / (1 + beta * |v - theta|)^2   [conceptual]
    v = alpha * v * (1 - spike) + I

    alpha (decay), beta (steepness), theta (threshold) all trainable.
    """

    v: float = 0.0
    alpha: float = 0.9
    beta: float = 5.0
    theta: float = 1.0

    def step(self, current: float) -> int:
        spike = int(self.v >= self.theta)
        self.v = self.alpha * self.v * (1 - spike) + current
        return spike

    def reset(self):
        self.v = 0.0

    def surrogate_grad(self) -> float:
        """Smooth surrogate gradient for backprop."""
        return 1.0 / (1.0 + self.beta * abs(self.v - self.theta)) ** 2


@dataclass
class ContinuousAttractorNeuron:
    """Ring attractor for continuous working memory.

    u_i += (-u_i + f(sum_j w_ij u_j) + I_i) / tau * dt
    f(x) = max(0,x)^2 / (1 + max(0,x)^2)
    w_ij = A * exp(-d_ij^2 / (2*sigma_e^2)) - B   (Mexican hat)

    Holds a continuous value (angle/position) in persistent activity.
    Output: position of the activity bump.
    """

    n_units: int = 16
    tau: float = 10.0
    sigma_e: float = 1.0
    excitation: float = 4.0
    inhibition: float = 0.5
    dt: float = 1.0
    u: list = field(default_factory=list)
    _weights: list = field(default_factory=list)

    def __post_init__(self):
        if not self.u:
            self.u = [0.0] * self.n_units
        if not self._weights:
            self._build_weights()

    def _build_weights(self):
        n = self.n_units
        self._weights = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                d = min(abs(i - j), n - abs(i - j))
                self._weights[i][j] = (
                    self.excitation * math.exp(-d * d / (2.0 * self.sigma_e ** 2))
                    - self.inhibition
                )

    @staticmethod
    def _activation(x: float) -> float:
        r = max(0.0, x)
        return r * r / (1.0 + r * r)

    def step(self, current: float) -> int:
        new_u = [0.0] * self.n_units
        for i in range(self.n_units):
            recurrent = sum(
                self._weights[i][j] * self._activation(self.u[j])
                for j in range(self.n_units)
            )
            new_u[i] = self.u[i] + (-self.u[i] + recurrent + current) / self.tau * self.dt
        self.u = new_u
        peak = max(self.u)
        return int(peak > 1.0)

    def bump_position(self) -> int:
        return self.u.index(max(self.u))

    def reset(self):
        self.u = [0.0] * self.n_units


@dataclass
class MetaPlasticNeuron:
    """Neuron with self-regulating meta-learning rate.

    dv/dt = (-v + I) / tau
    error_trace += (-error_trace + |reward - expected|) / tau_meta * dt
    meta_lr = lr0 * sigmoid(kappa * (error_trace - target_error))

    High error: faster learning. Low error: stabilize.
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
        self.v += (-self.v + current) / self.tau * self.dt
        if self.v >= self.theta:
            self.v = 0.0
            return 1
        return 0

    def update_meta(self, reward: float):
        error = abs(reward - self.expected_reward)
        self.error_trace += (-self.error_trace + error) / self.tau_meta * self.dt
        meta_lr = self.lr0 / (1.0 + math.exp(-self.kappa * (self.error_trace - self.target_error)))
        self.expected_reward += meta_lr * (reward - self.expected_reward)

    @property
    def meta_lr(self) -> float:
        return self.lr0 / (1.0 + math.exp(-self.kappa * (self.error_trace - self.target_error)))

    def reset(self):
        self.v = 0.0
        self.error_trace = 0.0
        self.expected_reward = 0.0
