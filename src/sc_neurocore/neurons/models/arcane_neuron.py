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
import math
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
        if not math.isfinite(current):
            raise ValueError("ArcaneNeuron current must be finite")
        self._validate_runtime_state()

        old_v_fast = float(self.v_fast)
        old_v_work = float(self.v_work)
        old_v_deep = float(self.v_deep)
        old_spike_history = list(self._spike_history)
        old_novelty_history = list(self._novelty_history)

        # Self-referential metrics
        spike_rate = float(sum(old_spike_history) / len(old_spike_history))
        confidence = 1.0 - float(np.mean(old_novelty_history))

        # Attention gate
        gate_input = (
            self.w_gate[0] * current
            + self.w_gate[1] * old_v_fast
            + self.w_gate[2] * old_v_work
            + self.w_gate[3] * confidence
        )
        gate = self._sigmoid(float(gate_input))
        i_eff = gate * current

        # Fast compartment: exact first-order relaxation to a constant drive
        # during this step.
        fast_drive = i_eff - self.w_inh * spike_rate
        next_v_fast_continuous = self._exact_relaxation(
            old_v_fast, fast_drive, self.dt, self.tau_fast
        )
        self._require_finite_candidate(next_v_fast_continuous, "fast compartment")

        # Prediction error (self-modeling)
        prediction = float(
            self.w_pred[0] * next_v_fast_continuous
            + self.w_pred[1] * old_v_work
            + self.w_pred[2] * old_v_deep
        )
        surprise = abs(next_v_fast_continuous - prediction)
        novelty = self._sigmoid(self.kappa * (surprise - self.surprise_baseline))

        # Effective threshold: deep state + confidence modulate
        eff_threshold = (
            self.theta * (1.0 + self.gamma * old_v_deep) * (1.0 - self.delta_conf * confidence)
        )
        self._require_finite_candidate(eff_threshold, "effective threshold")
        eff_threshold = max(eff_threshold, 0.1)

        # Spike decision
        spike = 1 if next_v_fast_continuous >= eff_threshold else 0
        accepted_v_fast = 0.0 if spike else next_v_fast_continuous

        # Working memory: exact relaxation to the spike-gated drive.
        work_drive = self.alpha_w * next_v_fast_continuous if spike else 0.0
        next_v_work = self._exact_relaxation(old_v_work, work_drive, self.dt, self.tau_work)
        self._require_finite_candidate(next_v_work, "working compartment")

        # Deep compartment: exact relaxation to novelty-gated working memory.
        deep_drive = self.alpha_d * next_v_work * novelty
        next_v_deep = self._exact_relaxation(old_v_deep, deep_drive, self.dt, self.tau_deep)
        self._require_finite_candidate(next_v_deep, "deep compartment")

        # Meta-learning: update predictor weights toward reducing surprise.
        meta_lr = self.lr_base * (1.0 + self.eta * novelty)
        error = accepted_v_fast - prediction
        next_w_pred = np.array(self.w_pred, dtype=float, copy=True)
        next_w_pred[0] += meta_lr * error * accepted_v_fast
        next_w_pred[1] += meta_lr * error * next_v_work
        next_w_pred[2] += meta_lr * error * next_v_deep
        norm = float(np.linalg.norm(next_w_pred))
        if not math.isfinite(norm):
            raise ValueError("ArcaneNeuron predictor candidate must remain finite")
        if norm > 0.0:
            next_w_pred /= norm

        next_novelty_history = old_novelty_history
        next_novelty_history[self._nov_idx % len(next_novelty_history)] = novelty
        next_spike_history = old_spike_history
        next_spike_history[self._hist_idx % len(next_spike_history)] = spike

        self.v_fast = accepted_v_fast
        self.v_work = next_v_work
        self.v_deep = next_v_deep
        self._prediction = prediction
        self._surprise = surprise
        self._novelty = novelty
        self._confidence = confidence
        self._novelty_history = next_novelty_history
        self._nov_idx += 1
        self._identity_drift += abs(next_v_deep - old_v_deep)
        self.w_pred = next_w_pred
        self._spike_history = next_spike_history
        self._hist_idx += 1
        self._total_steps += 1

        return spike

    @staticmethod
    def _exact_relaxation(state: float, steady_state: float, dt: float, tau: float) -> float:
        decay = math.exp(-dt / tau)
        return decay * state + (1.0 - decay) * steady_state

    @staticmethod
    def _sigmoid(x: float) -> float:
        if not math.isfinite(x):
            return 1.0 if x > 0.0 else 0.0
        if x >= 0.0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    @staticmethod
    def _require_finite_candidate(value: float, label: str) -> None:
        if not math.isfinite(value):
            raise ValueError(f"ArcaneNeuron {label} candidate must remain finite")

    def _validate_runtime_state(self) -> None:
        for name in (
            "v_fast",
            "v_work",
            "v_deep",
            "alpha_w",
            "alpha_d",
            "theta",
            "gamma",
            "delta_conf",
            "kappa",
            "surprise_baseline",
            "lr_base",
            "eta",
            "_prediction",
            "_surprise",
            "_novelty",
            "_confidence",
            "_identity_drift",
            "w_inh",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"ArcaneNeuron {name} must be finite")
        for name in ("tau_fast", "tau_work", "tau_deep", "dt"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"ArcaneNeuron {name} must be finite and positive")
        if self.theta <= 0.0:
            raise ValueError("ArcaneNeuron theta must be finite and positive")
        if self.alpha_w < 0.0 or self.alpha_d < 0.0 or self.lr_base < 0.0 or self.w_inh < 0.0:
            raise ValueError("ArcaneNeuron coupling and learning rates must be non-negative")
        w_gate = np.asarray(self.w_gate, dtype=float)
        w_pred = np.asarray(self.w_pred, dtype=float)
        if w_gate.shape != (4,) or not np.all(np.isfinite(w_gate)):
            raise ValueError("ArcaneNeuron w_gate must be a finite 4-vector")
        if w_pred.shape != (3,) or not np.all(np.isfinite(w_pred)):
            raise ValueError("ArcaneNeuron w_pred must be a finite 3-vector")
        if len(self._spike_history) == 0 or len(self._novelty_history) == 0:
            raise ValueError("ArcaneNeuron history buffers must be non-empty")
        if any(spike not in (0, 1) for spike in self._spike_history):
            raise ValueError("ArcaneNeuron spike history must contain binary values")
        if not all(math.isfinite(float(value)) for value in self._novelty_history):
            raise ValueError("ArcaneNeuron novelty history must be finite")
        if self._hist_idx < 0 or self._nov_idx < 0 or self._total_steps < 0:
            raise ValueError("ArcaneNeuron history counters must be non-negative")

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
