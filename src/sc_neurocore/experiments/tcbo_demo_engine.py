# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — TCBO Consciousness Detection Demo Engine

from __future__ import annotations
from typing import Any, Optional

"""
TCBO Consciousness Detection Demo Engine
==========================================

Self-contained demo proving consciousness boundary detection via
persistent homology proxy. Generates synthetic multichannel EEG
(Kuramoto oscillators), detects consciousness gate transitions,
and demonstrates PI controller recovery.

5 Scenarios:
    healthy_awake  - high coupling → high R → p_h1 > threshold
    anesthesia     - coupling drops + noise → p_h1 falls
    meditation     - alpha coherence sustained
    sleep_onset    - gradual coupling decay
    recovery       - PI controller restores kappa

"""


import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

import numpy as np

from sc_neurocore.scpn.params import OMEGA_N, build_knm_matrix

logger = logging.getLogger(__name__)


def _compute_order_parameter(theta: np.ndarray[Any, Any]) -> float:
    """Kuramoto order parameter R = |<e^(i*theta)>|."""
    z = np.mean(np.exp(1j * theta))
    return float(np.abs(z))


def _compute_p_h1_lightweight(
    phase_history: np.ndarray[Any, Any],
    tau_h1: float = 0.72,
    beta: float = 8.0,
) -> float:
    """Lightweight p_h1 proxy using phase-locking statistics.

    Computes average PLV across pairs from recent history,
    then applies logistic squash.
    """
    if phase_history.shape[0] < 10:
        return 0.0

    recent = phase_history[-50:]
    N = recent.shape[1]

    # Pairwise PLV (sample a subset of pairs)
    plvs = []
    rng = np.random.RandomState(0)
    n_pairs = min(30, N * (N - 1) // 2)
    for _ in range(n_pairs):
        i, j = rng.randint(0, N, 2)
        if i == j:
            continue
        diff = recent[:, i] - recent[:, j]
        plv = float(np.abs(np.mean(np.exp(1j * diff))))
        plvs.append(plv)

    if not plvs:
        return 0.0

    mean_plv = np.mean(plvs)
    # Logistic squash centered at tau_h1
    p_h1 = float(1.0 / (1.0 + np.exp(-beta * (mean_plv - tau_h1 + 0.3))))
    return float(np.clip(p_h1, 0.0, 1.0))


# ── Scenarios ──────────────────────────────────────────────────────────


class ScenarioName(str, Enum):
    HEALTHY_AWAKE = "healthy_awake"
    ANESTHESIA = "anesthesia"
    MEDITATION = "meditation"
    SLEEP_ONSET = "sleep_onset"
    RECOVERY = "recovery"


@dataclass
class ScenarioConfig:
    name: str
    description: str
    duration_s: float = 10.0
    K_scale: float = 1.0
    noise_amplitude: float = 0.3
    use_controller: bool = False
    phase_scramble: bool = False
    alpha_boost: float = 0.0
    coupling_decay_rate: float = 0.0


SCENARIOS: Dict[ScenarioName, ScenarioConfig] = {
    ScenarioName.HEALTHY_AWAKE: ScenarioConfig(
        name="healthy_awake",
        description="Normal waking: strong coupling → high coherence → gate OPEN",
        duration_s=10.0,
        K_scale=1.5,
        noise_amplitude=0.2,
    ),
    ScenarioName.ANESTHESIA: ScenarioConfig(
        name="anesthesia",
        description="Anesthesia: coupling drops 90%, noise 10x → gate CLOSED",
        duration_s=10.0,
        K_scale=0.1,
        noise_amplitude=2.0,
        phase_scramble=True,
    ),
    ScenarioName.MEDITATION: ScenarioConfig(
        name="meditation",
        description="Meditation: alpha coherence sustained → gate OPEN",
        duration_s=10.0,
        K_scale=1.2,
        noise_amplitude=0.15,
        alpha_boost=2.0,
    ),
    ScenarioName.SLEEP_ONSET: ScenarioConfig(
        name="sleep_onset",
        description="Sleep onset: gradual coupling decay → p_h1 declines",
        duration_s=15.0,
        K_scale=1.0,
        noise_amplitude=0.3,
        coupling_decay_rate=0.001,
    ),
    ScenarioName.RECOVERY: ScenarioConfig(
        name="recovery",
        description="Recovery: start suppressed, PI controller restores kappa",
        duration_s=15.0,
        K_scale=0.3,
        noise_amplitude=1.0,
        use_controller=True,
        phase_scramble=True,
    ),
}


# ── Synthetic EEG Generator ───────────────────────────────────────────


class SyntheticEEGGenerator:
    """Configurable Kuramoto oscillator network for synthetic EEG."""

    def __init__(
        self,
        N: int = 16,
        dt: float = 0.001,
        seed: int = 42,
    ):
        self.N = N
        self.dt = dt
        self._rng = np.random.RandomState(seed)
        self._seed = seed

        omega = OMEGA_N[:N] if N <= 16 else np.tile(OMEGA_N, (N // 16 + 1))[:N]
        self.omega = omega.copy()
        self._K_base = build_knm_matrix(N)
        self.K = self._K_base.copy()
        self.theta = self._rng.uniform(0, 2 * np.pi, N)
        self.noise_amplitude = 0.3
        self._step_count = 0

    def set_coupling_scale(self, scale: float) -> None:
        self.K = self._K_base * scale

    def apply_anesthesia(self, strength: float = 0.9) -> None:
        self.K *= 1.0 - strength
        self.theta = self._rng.uniform(0, 2 * np.pi, self.N)
        self.noise_amplitude *= 10.0

    def apply_alpha_boost(self, factor: float = 2.0) -> None:
        if self.N >= 3:
            self.K[1, :] *= factor
            self.K[:, 1] *= factor
            np.fill_diagonal(self.K, 0)

    def apply_coupling_decay(self, rate: float) -> None:
        self.K *= 1.0 - rate

    def step(self, perturbation: Optional[np.ndarray[Any, Any]] = None) -> np.ndarray[Any, Any]:
        """One Kuramoto timestep. Returns phases in [0, 2pi)."""
        dtheta = self.omega.copy()

        # Kuramoto coupling: Σ K_nm sin(θ_m - θ_n)
        for n in range(self.N):
            coupling = 0.0
            for m in range(self.N):
                if m != n:
                    coupling += self.K[n, m] * np.sin(self.theta[m] - self.theta[n])
            dtheta[n] += coupling

        # Noise
        dtheta += self.noise_amplitude * self._rng.randn(self.N)

        # External perturbation
        if perturbation is not None:
            dtheta += perturbation

        self.theta = (self.theta + dtheta * self.dt) % (2 * np.pi)
        self._step_count += 1
        return self.theta.copy()

    def run(self, n_steps: int) -> np.ndarray[Any, Any]:
        """Run n_steps, return (n_steps, N) history."""
        history = np.zeros((n_steps, self.N))
        for i in range(n_steps):
            history[i] = self.step()
        return history

    def get_order_parameter(self) -> float:
        return _compute_order_parameter(self.theta)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        self.theta = self._rng.uniform(0, 2 * np.pi, self.N)
        self.K = self._K_base.copy()
        self.noise_amplitude = 0.3
        self._step_count = 0


# ── PI Controller ──────────────────────────────────────────────────────


class TCBOController:
    """PI controller for gap-junction coupling kappa."""

    def __init__(
        self,
        tau_h1: float = 0.72,
        Kp: float = 2.0,
        Ki: float = 0.5,
        kappa_min: float = 0.1,
        kappa_max: float = 5.0,
    ):
        self.tau_h1 = tau_h1
        self.Kp = Kp
        self.Ki = Ki
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self._integral = 0.0

    def step(self, p_h1: float, kappa: float, dt: float) -> float:
        """Compute new kappa from consciousness deficit."""
        error = max(0.0, self.tau_h1 - p_h1)
        self._integral += error * dt
        # Anti-windup
        self._integral = np.clip(self._integral, 0, 10.0)
        delta = self.Kp * error + self.Ki * self._integral
        new_kappa = kappa + delta * dt
        return float(np.clip(new_kappa, self.kappa_min, self.kappa_max))

    def reset(self) -> None:
        self._integral = 0.0


# ── Demo Snapshot ──────────────────────────────────────────────────────


@dataclass
class TCBODemoSnapshot:
    """Per-step snapshot of the TCBO demo state."""

    step: int = 0
    time_s: float = 0.0
    phases: List[float] = field(default_factory=list)
    R_global: float = 0.0
    p_h1: float = 0.0
    gate_open: bool = False
    is_conscious: bool = False
    kappa: float = 1.0
    has_tcbo: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "time_s": round(self.time_s, 4),
            "phases": [round(p, 4) for p in self.phases],
            "R_global": round(self.R_global, 4),
            "p_h1": round(self.p_h1, 4),
            "gate_open": self.gate_open,
            "is_conscious": self.is_conscious,
            "kappa": round(self.kappa, 4),
            "has_tcbo": self.has_tcbo,
        }


# ── Demo Engine ────────────────────────────────────────────────────────


class TCBODemoEngine:
    """Orchestrates TCBO consciousness detection scenarios."""

    TAU_H1 = 0.72

    def __init__(self, N: int = 16, dt: float = 0.001, seed: int = 42):
        self.N = N
        self.dt = dt
        self._seed = seed
        self.gen = SyntheticEEGGenerator(N=N, dt=dt, seed=seed)
        self.controller = TCBOController(tau_h1=self.TAU_H1)

        self.p_h1 = 0.0
        self.kappa = 1.0
        self.is_running = False
        self._current_scenario: Optional[str] = None
        self._scenario_cfg: Optional[ScenarioConfig] = None
        self._step_count = 0
        self._max_steps = 0
        self._phase_history: List[np.ndarray[Any, Any]] = []
        self._snapshots: List[TCBODemoSnapshot] = []

    def get_scenarios(self) -> dict[str, dict[str, Any]]:
        return {
            name.value: {
                "name": cfg.name,
                "description": cfg.description,
                "duration_s": cfg.duration_s,
            }
            for name, cfg in SCENARIOS.items()
        }

    def start_scenario(self, name: str) -> dict[str, Any]:
        """Initialize and start a named scenario."""
        try:
            scenario_name = ScenarioName(name)
        except ValueError:
            raise ValueError(
                f"Unknown scenario: {name}. Available: {[s.value for s in ScenarioName]}"
            )

        cfg = SCENARIOS[scenario_name]
        self._current_scenario = name
        self._scenario_cfg = cfg

        # Reset generator
        self.gen.reset(seed=self._seed)
        self.gen.set_coupling_scale(cfg.K_scale)
        self.gen.noise_amplitude = cfg.noise_amplitude

        if cfg.phase_scramble:
            self.gen.theta = np.random.RandomState(self._seed + 1).uniform(0, 2 * np.pi, self.N)

        if cfg.alpha_boost > 0:
            self.gen.apply_alpha_boost(cfg.alpha_boost)

        self.controller.reset()
        self.kappa = cfg.K_scale
        self.p_h1 = 0.0
        self._step_count = 0
        self._max_steps = int(cfg.duration_s / self.dt)
        self._phase_history.clear()
        self._snapshots.clear()
        self.is_running = True

        return {"scenario": name, "max_steps": self._max_steps, "dt": self.dt}

    def step(self) -> TCBODemoSnapshot:
        """Advance one timestep."""
        if not self.is_running:
            raise RuntimeError("No scenario running")

        cfg = self._scenario_cfg

        # Apply coupling decay if configured
        if cfg and cfg.coupling_decay_rate > 0:
            self.gen.apply_coupling_decay(cfg.coupling_decay_rate)

        # Kuramoto step
        phases = self.gen.step()
        self._phase_history.append(phases)

        # Keep bounded history
        if len(self._phase_history) > 200:
            self._phase_history = self._phase_history[-100:]

        # Compute observables
        R = self.gen.get_order_parameter()
        history_arr = np.array(self._phase_history)
        self.p_h1 = _compute_p_h1_lightweight(history_arr, self.TAU_H1)
        gate_open = self.p_h1 > self.TAU_H1

        # PI controller
        if cfg and cfg.use_controller:
            new_kappa = self.controller.step(self.p_h1, self.kappa, self.dt)
            if new_kappa > self.kappa:
                self.gen.set_coupling_scale(new_kappa)
            self.kappa = new_kappa

        self._step_count += 1
        if self._step_count >= self._max_steps:
            self.is_running = False

        snap = TCBODemoSnapshot(
            step=self._step_count,
            time_s=self._step_count * self.dt,
            phases=phases.tolist(),
            R_global=R,
            p_h1=self.p_h1,
            gate_open=gate_open,
            is_conscious=gate_open,
            kappa=self.kappa,
            has_tcbo=False,
        )
        self._snapshots.append(snap)
        return snap

    def run_scenario(
        self,
        name: str,
        duration_s: Optional[float] = None,
        subsample: int = 100,
    ) -> List[TCBODemoSnapshot]:
        """Run a full scenario, returning subsampled snapshots."""
        self.start_scenario(name)
        if duration_s is not None:
            self._max_steps = int(duration_s / self.dt)

        results = []
        for i in range(self._max_steps):
            snap = self.step()
            if i % subsample == 0:
                results.append(snap)

        return results

    def get_state(self) -> dict[str, Any]:
        return {
            "running": self.is_running,
            "scenario": self._current_scenario,
            "step": self._step_count,
            "p_h1": round(self.p_h1, 4),
            "kappa": round(self.kappa, 4),
            "R_global": round(self.gen.get_order_parameter(), 4),
            "gate_open": self.p_h1 > self.TAU_H1,
        }

    def get_history(self, last_n: int = 100) -> List[dict[str, Any]]:
        return [s.to_dict() for s in self._snapshots[-last_n:]]

    def reset(self) -> None:
        self.gen.reset(seed=self._seed)
        self.controller.reset()
        self.p_h1 = 0.0
        self.kappa = 1.0
        self.is_running = False
        self._current_scenario = None
        self._step_count = 0
        self._phase_history.clear()
        self._snapshots.clear()


# ── Singleton ──────────────────────────────────────────────────────────

_engine: Optional[TCBODemoEngine] = None


def get_tcbo_demo_engine() -> TCBODemoEngine:
    global _engine
    if _engine is None:
        _engine = TCBODemoEngine()
    return _engine


def reset_tcbo_demo_engine() -> None:
    global _engine
    _engine = None
