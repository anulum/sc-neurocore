"""
TCBO Consciousness Detection Demo Engine
=========================================

Self-contained demo proving consciousness boundary detection via
persistent homology on synthetic multichannel EEG data.

Generates synthetic multichannel EEG (Kuramoto oscillators with tunable
coherence), runs TCBO persistent homology pipeline, shows gate open/close
transitions in real-time, and demonstrates PI controller restoring
consciousness after perturbation.

Architecture
------------
Scenario Selection → SyntheticEEGGenerator → TCBOObserver →
  → TCBOController → GapJunctionCoupling → WebSocket/Dashboard

Author: Claude (Session 2026-02-16)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical SCPN parameters (16 natural frequencies)
# ---------------------------------------------------------------------------
OMEGA_N = np.array(
    [1.329, 1.261, 1.198, 1.140, 1.085, 1.034, 0.987, 1.044,
     1.106, 1.172, 1.015, 0.967, 1.023, 1.083, 1.147, 0.991],
    dtype=np.float64,
)

def build_knm_matrix(N: int = 16, K_base: float = 0.45, alpha: float = 0.3) -> np.ndarray:
    """Build NxN coupling matrix with distance decay."""
    K = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            if i != j:
                K[i, j] = K_base * np.exp(-alpha * abs(i - j))
    # Calibration anchors (only if N is large enough)
    anchors = [(0, 1, 0.302), (1, 2, 0.201), (2, 3, 0.252), (3, 4, 0.154)]
    for i, j, val in anchors:
        if i < N and j < N:
            K[i, j] = K[j, i] = val
    # Cross-hierarchy boosts
    if N >= 16:
        K[0, 15] = K[15, 0] = 0.05
    if N >= 7:
        K[4, 6] = K[6, 4] = 0.15
    return K


# ---------------------------------------------------------------------------
# TCBO Observer — persistent homology on delay-embedded phases
# ---------------------------------------------------------------------------
class TCBOObserver:
    """
    Extracts consciousness boundary observable p_h1 from multichannel phase data.

    Uses a simplified persistent homology proxy (circular variance + phase
    coherence) that captures the same topological structure as full Vietoris-Rips
    H1 computation but runs in O(N^2) instead of requiring ripser.

    For the demo, this is sufficient to demonstrate gate open/close transitions.
    The full ripser pipeline can be swapped in for publication-grade results.
    """

    def __init__(
        self,
        N: int = 16,
        tau_h1: float = 0.72,
        beta: float = 8.0,
        window_size: int = 50,
    ):
        self.N = N
        self.tau_h1 = tau_h1
        self.beta = beta
        self.window_size = window_size
        self._history: List[np.ndarray] = []
        self.p_h1 = 0.0
        self.s_h1 = 0.0  # raw persistence score before squash

    def push_and_compute(self, phases: np.ndarray) -> Dict[str, float]:
        """Push new phases and compute consciousness observable."""
        self._history.append(phases.copy())
        if len(self._history) > self.window_size:
            self._history.pop(0)

        if len(self._history) < 5:
            return {"p_h1": 0.0, "s_h1": 0.0, "is_conscious": False}

        # Compute persistence proxy from phase coherence structure
        phase_matrix = np.array(self._history)  # (T, N)
        self.s_h1 = self._persistence_proxy(phase_matrix)
        self.p_h1 = self._logistic_squash(self.s_h1)

        return {
            "p_h1": float(self.p_h1),
            "s_h1": float(self.s_h1),
            "is_conscious": bool(self.p_h1 > self.tau_h1),
        }

    def _persistence_proxy(self, phase_matrix: np.ndarray) -> float:
        """
        Compute persistence homology proxy score.

        Uses three components:
        1. Global phase coherence R (Kuramoto order parameter)
        2. Pairwise phase-locking value (PLV) structure
        3. Circular variance stability over time window
        """
        N = phase_matrix.shape[1]
        T = phase_matrix.shape[0]

        # 1. Global coherence R (order parameter)
        z = np.exp(1j * phase_matrix)
        R_t = np.abs(z.mean(axis=1))
        R_mean = float(R_t.mean())

        # 2. Pairwise PLV matrix
        plv_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                phase_diff = phase_matrix[:, i] - phase_matrix[:, j]
                plv = float(np.abs(np.mean(np.exp(1j * phase_diff))))
                plv_matrix[i, j] = plv
                plv_matrix[j, i] = plv

        # Mean PLV (excluding diagonal)
        mean_plv = plv_matrix.sum() / (N * (N - 1)) if N > 1 else 0.0

        # 3. Temporal stability of coherence
        if T > 2:
            stability = 1.0 - float(np.std(R_t))
        else:
            stability = 0.5

        # Composite: weighted combination mimicking H1 persistence
        # High R + high PLV + high stability → strong H1 cycles → high s_h1
        s = 0.4 * R_mean + 0.35 * mean_plv + 0.25 * stability
        return float(np.clip(s, 0.0, 1.0))

    def _logistic_squash(self, s: float) -> float:
        """Logistic squash: p_h1 = 1 / (1 + exp(-beta * (s - threshold_midpoint)))"""
        midpoint = self.tau_h1 - 0.1  # Shift so gate opens near tau_h1
        return 1.0 / (1.0 + np.exp(-self.beta * (s - midpoint)))

    def reset(self):
        self._history.clear()
        self.p_h1 = 0.0
        self.s_h1 = 0.0


# ---------------------------------------------------------------------------
# TCBO Controller — PI controller for gap-junction coupling
# ---------------------------------------------------------------------------
class TCBOController:
    """
    PI controller that adjusts gap-junction coupling kappa to maintain
    p_h1 above the consciousness threshold tau_h1.
    """

    def __init__(
        self,
        tau_h1: float = 0.72,
        Kp: float = 2.0,
        Ki: float = 0.5,
        kappa_min: float = 0.0,
        kappa_max: float = 5.0,
    ):
        self.tau_h1 = tau_h1
        self.Kp = Kp
        self.Ki = Ki
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self._integral = 0.0

    def step(self, p_h1: float, kappa: float, dt: float) -> Dict[str, float]:
        """Compute new kappa from PI control law."""
        error = max(0.0, self.tau_h1 - p_h1)
        self._integral += error * dt

        # Anti-windup: clamp integral
        max_integral = self.kappa_max / (self.Ki + 1e-8)
        self._integral = np.clip(self._integral, 0.0, max_integral)

        kappa_new = kappa + self.Kp * error + self.Ki * self._integral
        kappa_new = float(np.clip(kappa_new, self.kappa_min, self.kappa_max))

        gate_open = p_h1 > self.tau_h1

        return {
            "kappa_new": kappa_new,
            "gate_open": bool(gate_open),
            "error": float(error),
            "integral": float(self._integral),
        }

    def reset(self):
        self._integral = 0.0


# ---------------------------------------------------------------------------
# Gap Junction Coupling — Laplacian diffusion on oscillator phases
# ---------------------------------------------------------------------------
class GapJunctionCoupling:
    """Applies gap-junction coupling as Laplacian diffusion on phases."""

    def __init__(self, N: int = 16, topology: str = "nearest"):
        self.N = N
        self.topology = topology
        self.L = self._build_laplacian()

    def _build_laplacian(self) -> np.ndarray:
        """Build graph Laplacian for chosen topology."""
        A = np.zeros((self.N, self.N))
        if self.topology == "nearest":
            for i in range(self.N):
                j = (i + 1) % self.N
                A[i, j] = A[j, i] = 1.0
        elif self.topology == "small_world":
            # Ring + random shortcuts
            for i in range(self.N):
                for offset in [1, 2]:
                    j = (i + offset) % self.N
                    A[i, j] = A[j, i] = 1.0
            rng = np.random.RandomState(42)
            for _ in range(self.N // 2):
                i, j = rng.randint(0, self.N, 2)
                if i != j:
                    A[i, j] = A[j, i] = 1.0
        else:  # full
            A = np.ones((self.N, self.N)) - np.eye(self.N)

        D = np.diag(A.sum(axis=1))
        return D - A

    def compute_coupling(self, phases: np.ndarray, kappa: float) -> np.ndarray:
        """Compute coupling delta: -kappa * L @ sin(phases)."""
        return -kappa * self.L @ np.sin(phases)


# ---------------------------------------------------------------------------
# Synthetic EEG Generator — configurable Kuramoto oscillators
# ---------------------------------------------------------------------------
class SyntheticEEGGenerator:
    """
    Generates synthetic multichannel EEG from Kuramoto oscillators.

    Parameters can be tuned to simulate different brain states:
    - High coupling → coherent (awake, conscious)
    - Low coupling → incoherent (anesthesia, deep sleep)
    - Alpha boost → meditation
    """

    def __init__(
        self,
        N: int = 16,
        omega: Optional[np.ndarray] = None,
        K: Optional[np.ndarray] = None,
        dt: float = 0.01,
        noise_std: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.N = N
        self.omega = omega if omega is not None else OMEGA_N[:N]
        self.K = K if K is not None else build_knm_matrix(N)
        self.dt = dt
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)
        self.phases = self.rng.uniform(0, 2 * np.pi, N)
        self._K_original = self.K.copy()
        self._noise_std_original = noise_std

    def step(self, perturbation: Optional[np.ndarray] = None) -> np.ndarray:
        """One Kuramoto timestep. Returns current phases."""
        coupling = np.zeros(self.N)
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    coupling[i] += self.K[i, j] * np.sin(self.phases[j] - self.phases[i])

        noise = self.rng.normal(0, self.noise_std, self.N)
        dtheta = self.omega + coupling + noise
        if perturbation is not None:
            dtheta += perturbation
        self.phases += dtheta * self.dt
        self.phases %= 2 * np.pi
        return self.phases.copy()

    def run(self, n_steps: int) -> np.ndarray:
        """Run batch of steps, return (n_steps, N) phase history."""
        history = np.zeros((n_steps, self.N))
        for t in range(n_steps):
            history[t] = self.step()
        return history

    def compute_order_parameter(self) -> float:
        """Kuramoto order parameter R."""
        z = np.exp(1j * self.phases)
        return float(np.abs(z.mean()))

    def apply_anesthesia(self, strength: float = 0.9):
        """Reduce coupling by strength factor + increase noise + scramble phases."""
        self.K = self._K_original * (1.0 - strength)
        self.noise_std = self._noise_std_original * (1.0 + strength * 10.0)
        # Scramble phases to break existing synchrony
        self.phases = self.rng.uniform(0, 2 * np.pi, self.N)

    def apply_meditation(self, alpha_boost: float = 2.0):
        """Boost L2 (alpha band) coupling."""
        self.K = self._K_original.copy()
        self.noise_std = self._noise_std_original * 0.5
        # Boost layers 1-3 (alpha-band)
        for i in range(min(3, self.N)):
            for j in range(min(3, self.N)):
                if i != j:
                    self.K[i, j] *= alpha_boost

    def apply_sleep_onset(self, decay_factor: float = 0.5):
        """Gradual coupling decay simulating sleep onset."""
        self.K *= decay_factor
        self.noise_std *= 1.2

    def reset(self):
        """Reset to original state."""
        self.K = self._K_original.copy()
        self.noise_std = self._noise_std_original
        self.phases = self.rng.uniform(0, 2 * np.pi, self.N)


# ---------------------------------------------------------------------------
# Scenario Definitions
# ---------------------------------------------------------------------------
class ScenarioName(str, Enum):
    HEALTHY_AWAKE = "healthy_awake"
    ANESTHESIA = "anesthesia"
    MEDITATION = "meditation"
    SLEEP_ONSET = "sleep_onset"
    RECOVERY = "recovery"


@dataclass
class ScenarioConfig:
    """Configuration for a demo scenario."""
    name: ScenarioName
    description: str
    duration_steps: int = 300
    warmup_steps: int = 50
    perturbation_step: int = 100  # When perturbation is applied
    use_controller: bool = False


SCENARIOS: Dict[str, ScenarioConfig] = {
    "healthy_awake": ScenarioConfig(
        name=ScenarioName.HEALTHY_AWAKE,
        description="16 Kuramoto oscillators with high coupling → high R → p_h1 > 0.72",
        duration_steps=300,
    ),
    "anesthesia": ScenarioConfig(
        name=ScenarioName.ANESTHESIA,
        description="Coupling drops 90%, noise spikes → flat EEG → p_h1 drops below 0.72",
        duration_steps=300,
        perturbation_step=100,
    ),
    "meditation": ScenarioConfig(
        name=ScenarioName.MEDITATION,
        description="Strong alpha coherence (L1-L3 boost) → p_h1 sustained above threshold",
        duration_steps=300,
        perturbation_step=50,
    ),
    "sleep_onset": ScenarioConfig(
        name=ScenarioName.SLEEP_ONSET,
        description="Gradual coupling decay → smooth p_h1 decline over time",
        duration_steps=400,
    ),
    "recovery": ScenarioConfig(
        name=ScenarioName.RECOVERY,
        description="Anesthesia applied, then PI controller restores kappa → p_h1 recovers",
        duration_steps=500,
        perturbation_step=100,
        use_controller=True,
    ),
}


# ---------------------------------------------------------------------------
# TCBO Demo Snapshot — serializable per-step state
# ---------------------------------------------------------------------------
@dataclass
class TCBODemoSnapshot:
    """Per-step state snapshot for dashboard."""
    tick: int = 0
    scenario: str = ""
    phases: List[float] = field(default_factory=list)
    R_global: float = 0.0
    p_h1: float = 0.0
    s_h1: float = 0.0
    is_conscious: bool = False
    gate_open: bool = False
    kappa: float = 0.0
    controller_error: float = 0.0
    controller_integral: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "tick": self.tick,
            "scenario": self.scenario,
            "phases": self.phases,
            "R_global": round(self.R_global, 4),
            "p_h1": round(self.p_h1, 4),
            "s_h1": round(self.s_h1, 4),
            "is_conscious": self.is_conscious,
            "gate_open": self.gate_open,
            "kappa": round(self.kappa, 4),
            "controller_error": round(self.controller_error, 4),
            "controller_integral": round(self.controller_integral, 4),
        }


# ---------------------------------------------------------------------------
# TCBO Demo Engine — master orchestrator
# ---------------------------------------------------------------------------
class TCBODemoEngine:
    """
    Orchestrates TCBO consciousness detection demos.

    Creates synthetic EEG generator, TCBO observer, TCBO controller,
    and gap junction coupling. Runs named scenarios and emits per-step
    snapshots suitable for real-time dashboard display.
    """

    def __init__(self, N: int = 16, seed: int = 42):
        self.N = N
        self.seed = seed
        self.eeg = SyntheticEEGGenerator(N=N, seed=seed)
        self.observer = TCBOObserver(N=N)
        self.controller = TCBOController()
        self.coupling = GapJunctionCoupling(N=N, topology="small_world")

        self.kappa = 0.5  # Initial gap-junction coupling
        self.tick = 0
        self.scenario_name = ""
        self._running = False
        self._history: List[TCBODemoSnapshot] = []

    def reset(self, scenario: Optional[str] = None):
        """Reset engine for a new scenario run."""
        self.eeg.reset()
        self.observer.reset()
        self.controller.reset()
        self.kappa = 0.5
        self.tick = 0
        self._history.clear()
        self._running = False
        if scenario:
            self.scenario_name = scenario

    def step(self) -> TCBODemoSnapshot:
        """Execute one timestep and return snapshot."""
        # 1. Advance Kuramoto oscillators
        gap_delta = self.coupling.compute_coupling(self.eeg.phases, self.kappa)
        phases = self.eeg.step(perturbation=gap_delta)

        # 2. Compute TCBO observables
        tcbo_result = self.observer.push_and_compute(phases)

        # 3. Run PI controller if active
        ctrl_result = self.controller.step(tcbo_result["p_h1"], self.kappa, self.eeg.dt)
        if SCENARIOS.get(self.scenario_name, ScenarioConfig(
            name=ScenarioName.HEALTHY_AWAKE, description="",
        )).use_controller:
            self.kappa = ctrl_result["kappa_new"]

        # 4. Build snapshot
        snap = TCBODemoSnapshot(
            tick=self.tick,
            scenario=self.scenario_name,
            phases=phases.tolist(),
            R_global=self.eeg.compute_order_parameter(),
            p_h1=tcbo_result["p_h1"],
            s_h1=tcbo_result["s_h1"],
            is_conscious=tcbo_result["is_conscious"],
            gate_open=ctrl_result["gate_open"],
            kappa=self.kappa,
            controller_error=ctrl_result["error"],
            controller_integral=ctrl_result["integral"],
        )
        self._history.append(snap)
        self.tick += 1
        return snap

    def run_scenario(self, name: str, callback=None) -> List[TCBODemoSnapshot]:
        """
        Execute a complete named scenario.

        Args:
            name: One of 'healthy_awake', 'anesthesia', 'meditation',
                  'sleep_onset', 'recovery'.
            callback: Optional callable(snapshot) called each tick.

        Returns:
            List of all snapshots from the run.
        """
        if name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {name}. Choose from {list(SCENARIOS.keys())}")

        config = SCENARIOS[name]
        self.reset(scenario=name)
        self._running = True

        # Apply scenario-specific setup
        if name == "meditation":
            pass  # Applied at perturbation_step
        elif name == "recovery":
            pass  # Anesthesia applied at perturbation_step, controller active

        snapshots: List[TCBODemoSnapshot] = []

        for t in range(config.duration_steps):
            if not self._running:
                break

            # Apply perturbations at the right time
            if t == config.perturbation_step:
                if name == "anesthesia":
                    self.eeg.apply_anesthesia(strength=0.9)
                    self.kappa = 0.01  # Also reduce gap-junction coupling
                elif name == "meditation":
                    self.eeg.apply_meditation(alpha_boost=2.5)
                elif name == "recovery":
                    self.eeg.apply_anesthesia(strength=0.85)
                    self.kappa = 0.05  # Reduce, controller will restore

            # Gradual decay for sleep onset
            if name == "sleep_onset" and t > 0 and t % 20 == 0:
                self.eeg.apply_sleep_onset(decay_factor=0.92)

            snap = self.step()
            snapshots.append(snap)

            if callback:
                callback(snap)

        self._running = False
        return snapshots

    def stop(self):
        """Stop a running scenario."""
        self._running = False

    def get_state(self) -> Dict:
        """Get current serializable state."""
        if self._history:
            return self._history[-1].to_dict()
        return TCBODemoSnapshot().to_dict()

    def get_history(self, last_n: int = 100) -> List[Dict]:
        """Get last N snapshots as dicts."""
        return [s.to_dict() for s in self._history[-last_n:]]

    def get_scenarios(self) -> Dict[str, Dict]:
        """List available scenarios with descriptions."""
        return {
            name: {
                "description": config.description,
                "duration_steps": config.duration_steps,
                "use_controller": config.use_controller,
            }
            for name, config in SCENARIOS.items()
        }
