# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ArcaneNeuron: unified self-referential cognition model

"""ArcaneNeuron — the first neuron model designed for persistent identity.

No equivalent exists in any toolkit. Combines five coupled subsystems
in a single coherent ODE:

1. FAST (tau=5ms): spike timing, immediate sensory processing
2. WORKING (tau=200ms): working memory via sustained activity
3. DEEP (tau=10s): long-term context accumulation, personality drift
4. GATE: learned attention over inputs, modulated by confidence
5. PREDICTOR: forward model of own future state, fires on surprise

Core equations:

    gate = sigmoid(w_g @ [I, v_fast, v_work, confidence])
    I_eff = gate * I

    dv_fast/dt = (-v_fast + I_eff - w_inh * spike_history_rate) / tau_fast
    dv_work/dt = (-v_work + alpha_w * v_fast * spike) / tau_work
    dv_deep/dt = (-v_deep + alpha_d * v_work * novelty) / tau_deep

    prediction = w_pred @ [v_fast, v_work, v_deep]
    surprise = |v_fast - prediction|
    novelty = sigmoid(kappa * (surprise - surprise_baseline))

    confidence = 1 - mean(novelty_history)
    effective_threshold = theta * (1 + gamma * v_deep) * (1 - delta * confidence)

    meta_lr = lr_base * (1 + eta * novelty)  # learn more when surprised

    spike when v_fast >= effective_threshold

The deep compartment accumulates identity: it changes only when the
neuron encounters genuine novelty (prediction errors), not routine
input. Confidence modulates the threshold (confident = lower threshold
= faster responses) and the learning rate (uncertain = learn faster).

The gate prevents irrelevant input from reaching the fast compartment,
implementing selective attention as a neuronal property.

The predictor learns to forecast the neuron's own future fast-state.
When reality deviates from prediction, novelty rises, deep compartment
updates, and meta-learning rate increases. This is predictive
self-modeling — the neuron builds a model of itself.

Reference: Original design, Šotek & Arcane Sapience 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class ArcaneNeuron:
    """Unified self-referential cognition neuron for persistent identity."""

    # Fast compartment (spike timing)
    v_fast: float = 0.0
    tau_fast: float = 5.0
    # Working memory compartment
    v_work: float = 0.0
    tau_work: float = 200.0
    alpha_w: float = 0.3
    # Deep context compartment (identity accumulation)
    v_deep: float = 0.0
    tau_deep: float = 10000.0
    alpha_d: float = 0.05
    # Threshold
    theta: float = 1.0
    gamma: float = 0.2
    delta_conf: float = 0.3
    # Gate weights (4 inputs: I, v_fast, v_work, confidence)
    w_gate: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.1, 0.05, 0.05]))
    # Predictor weights (3 inputs: v_fast, v_work, v_deep)
    w_pred: np.ndarray = field(default_factory=lambda: np.array([0.6, 0.3, 0.1]))
    # Novelty detection
    kappa: float = 5.0
    surprise_baseline: float = 0.1
    # Meta-learning
    lr_base: float = 0.01
    eta: float = 2.0
    # Self-referential state
    _prediction: float = 0.0
    _surprise: float = 0.0
    _novelty: float = 0.0
    _confidence: float = 0.5
    _spike_history: list = field(default_factory=lambda: [0] * 50)
    _novelty_history: list = field(default_factory=lambda: [0.5] * 20)
    _hist_idx: int = 0
    _nov_idx: int = 0
    _total_steps: int = 0
    _identity_drift: float = 0.0
    # Inhibitory self-feedback
    w_inh: float = 0.3
    # Timestep
    dt: float = 1.0

    def step(self, current: float) -> int:
        # Self-referential metrics
        spike_rate = sum(self._spike_history) / len(self._spike_history)
        self._confidence = 1.0 - np.mean(self._novelty_history)

        # Attention gate
        gate_input = (
            self.w_gate[0] * current
            + self.w_gate[1] * self.v_fast
            + self.w_gate[2] * self.v_work
            + self.w_gate[3] * self._confidence
        )
        gate = 1.0 / (1.0 + np.exp(-gate_input))
        i_eff = gate * current

        # Fast compartment
        self.v_fast += (-self.v_fast + i_eff - self.w_inh * spike_rate) / self.tau_fast * self.dt

        # Prediction error (self-modeling)
        self._prediction = (
            self.w_pred[0] * self.v_fast
            + self.w_pred[1] * self.v_work
            + self.w_pred[2] * self.v_deep
        )
        self._surprise = abs(self.v_fast - self._prediction)
        self._novelty = 1.0 / (
            1.0 + np.exp(-self.kappa * (self._surprise - self.surprise_baseline))
        )

        # Update novelty history
        self._novelty_history[self._nov_idx % len(self._novelty_history)] = self._novelty
        self._nov_idx += 1

        # Effective threshold: deep state + confidence modulate
        eff_threshold = (
            self.theta
            * (1.0 + self.gamma * self.v_deep)
            * (1.0 - self.delta_conf * self._confidence)
        )
        eff_threshold = max(eff_threshold, 0.1)

        # Spike decision
        spike = 1 if self.v_fast >= eff_threshold else 0

        # Working memory: only updates on spike
        if spike:
            self.v_work += self.alpha_w * self.v_fast / self.tau_work * self.dt
            self.v_fast = 0.0

        # Working memory decay
        self.v_work += -self.v_work / self.tau_work * self.dt

        # Deep compartment: only updates on genuine novelty
        prev_deep = self.v_deep
        self.v_deep += (
            (-self.v_deep + self.alpha_d * self.v_work * self._novelty) / self.tau_deep * self.dt
        )
        self._identity_drift += abs(self.v_deep - prev_deep)

        # Meta-learning: update predictor weights toward reducing surprise
        meta_lr = self.lr_base * (1.0 + self.eta * self._novelty)
        error = self.v_fast - self._prediction
        self.w_pred[0] += meta_lr * error * self.v_fast
        self.w_pred[1] += meta_lr * error * self.v_work
        self.w_pred[2] += meta_lr * error * self.v_deep
        norm = np.linalg.norm(self.w_pred)
        if norm > 0:
            self.w_pred /= norm

        # Update spike history
        self._spike_history[self._hist_idx % len(self._spike_history)] = spike
        self._hist_idx += 1
        self._total_steps += 1

        return spike

    def reset(self) -> None:
        self.v_fast = 0.0
        self.v_work = 0.0
        # Deep compartment does NOT reset — it IS the identity
        self._prediction = 0.0
        self._surprise = 0.0
        self._novelty = 0.0
        self._spike_history = [0] * 50
        self._hist_idx = 0
        self._identity_drift = 0.0

    @property
    def identity_state(self) -> float:
        """The deep compartment value — the accumulated identity."""
        return self.v_deep

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def novelty(self) -> float:
        return self._novelty

    @property
    def identity_drift(self) -> float:
        """Cumulative absolute magnitude of identity mutation."""
        return self._identity_drift

    @property
    def meta_learning_rate(self) -> float:
        return self.lr_base * (1.0 + self.eta * self._novelty)

    def get_recent_pre_activity(self) -> float:
        """Get proxy for pre-synaptic activation (recent spike behavior)."""
        hist_ix = (self._hist_idx - 1) % max(1, len(self._spike_history))
        return float(self._spike_history[hist_ix])

    def get_state(self) -> dict:
        return {
            "v_fast": self.v_fast,
            "v_work": self.v_work,
            "v_deep": self.v_deep,
            "confidence": self._confidence,
            "novelty": self._novelty,
            "surprise": self._surprise,
            "prediction": self._prediction,
            "identity_drift": self._identity_drift,
            "meta_lr": self.meta_learning_rate,
            "total_steps": self._total_steps,
        }
