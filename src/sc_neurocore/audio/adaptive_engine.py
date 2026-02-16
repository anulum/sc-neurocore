"""
Adaptive Audio Engine — EVS → SSGF feedback loop
==================================================

The core feedback loop: EVS snapshots drive SSGF cost weight modulation
and audio parameter generation. Three session phases:

Phase 1 (0-2min):  Baseline discovery, gentle frequency sweep
Phase 2 (2-10min): Lock onto responsive frequency, fine-tune
Phase 3 (10min+):  Deepen entrainment, enable theurgic mode

Author: Claude (Session 2026-02-16)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from .evs_engine import EVSEngine, EVSConfig, EVSSnapshot
from .ssgf_engine import SSGFEngine, SSGFConfig, SSGFState
from .user_profile import UserProfile


class SessionPhase(str, Enum):
    DISCOVERY = "discovery"       # Phase 1: 0-2 min
    LOCK_ON = "lock_on"          # Phase 2: 2-10 min
    DEEPENING = "deepening"       # Phase 3: 10+ min


@dataclass
class AdaptiveConfig:
    """Adaptive audio engine configuration."""
    phase1_duration_s: float = 120.0   # Discovery phase
    phase2_duration_s: float = 480.0   # Lock-on phase
    evs_low_threshold: float = 40.0    # Below: increase sweep
    evs_mid_threshold: float = 70.0    # Above: start deepening
    evs_trend_window: int = 10         # Snapshots for trend calculation
    theurgic_R_threshold: float = 0.95 # R required for theurgic mode
    adaptation_rate: float = 0.1       # How fast to adapt SSGF params


@dataclass
class AdaptiveSnapshot:
    """Per-tick snapshot of the adaptive engine."""
    tick: int = 0
    session_phase: str = "discovery"
    evs_score: float = 0.0
    evs_trend: float = 0.0  # Positive = improving
    R_global: float = 0.0
    audio_params: Dict[str, float] = field(default_factory=dict)
    ssgf_state: Dict[str, float] = field(default_factory=dict)
    adaptations_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "tick": self.tick,
            "session_phase": self.session_phase,
            "evs_score": round(self.evs_score, 2),
            "evs_trend": round(self.evs_trend, 3),
            "R_global": round(self.R_global, 4),
            "audio_params": self.audio_params,
            "ssgf_state": self.ssgf_state,
            "adaptations_applied": self.adaptations_applied,
        }


@dataclass
class AdaptiveSessionReport:
    """Post-session analytics."""
    duration_s: float = 0.0
    total_ticks: int = 0
    evs_avg: float = 0.0
    evs_peak: float = 0.0
    evs_trend_final: float = 0.0
    time_verified_pct: float = 0.0
    phase_durations: Dict[str, float] = field(default_factory=dict)
    theurgic_time_pct: float = 0.0
    grade: str = "F"

    def to_dict(self) -> Dict:
        return {
            "duration_s": round(self.duration_s, 1),
            "total_ticks": self.total_ticks,
            "evs_avg": round(self.evs_avg, 2),
            "evs_peak": round(self.evs_peak, 2),
            "evs_trend_final": round(self.evs_trend_final, 3),
            "time_verified_pct": round(self.time_verified_pct, 1),
            "phase_durations": self.phase_durations,
            "theurgic_time_pct": round(self.theurgic_time_pct, 1),
            "grade": self.grade,
        }


class AdaptiveAudioEngine:
    """
    Core adaptive feedback loop: EVS → SSGF → audio parameters.

    Usage:
        engine = AdaptiveAudioEngine(ssgf, evs, profile)
        engine.start_session()
        while running:
            # Add EEG samples to EVS externally
            snapshot = engine.on_evs_update(evs.compute())
            # Use snapshot.audio_params for audio synthesis
        report = engine.get_session_report()
    """

    def __init__(
        self,
        ssgf: SSGFEngine,
        evs: EVSEngine,
        profile: Optional[UserProfile] = None,
        config: Optional[AdaptiveConfig] = None,
    ):
        self.ssgf = ssgf
        self.evs = evs
        self.profile = profile or UserProfile()
        self.config = config or AdaptiveConfig()

        self._tick = 0
        self._start_time = 0.0
        self._running = False
        self._evs_history: deque = deque(maxlen=100)
        self._snapshots: List[AdaptiveSnapshot] = []
        self._verified_count = 0
        self._theurgic_count = 0

    @property
    def session_phase(self) -> SessionPhase:
        """Determine current session phase based on elapsed time."""
        elapsed = self._tick  # Using ticks as proxy for time
        if elapsed < self.config.phase1_duration_s:
            return SessionPhase.DISCOVERY
        elif elapsed < self.config.phase1_duration_s + self.config.phase2_duration_s:
            return SessionPhase.LOCK_ON
        else:
            return SessionPhase.DEEPENING

    def start_session(self):
        """Initialize a new adaptive session."""
        self._tick = 0
        self._start_time = time.monotonic()
        self._running = True
        self._evs_history.clear()
        self._snapshots.clear()
        self._verified_count = 0
        self._theurgic_count = 0

        # Apply profile-based SSGF overrides
        overrides = self.profile.get_ssgf_config_overrides()
        self.ssgf.update_config(**overrides)

    def on_evs_update(self, evs_snapshot: EVSSnapshot) -> AdaptiveSnapshot:
        """
        Process EVS update and adapt SSGF parameters.

        This is the core feedback loop called each time EVS computes a new score.
        """
        self._evs_history.append(evs_snapshot.evs_score)
        if evs_snapshot.is_verified:
            self._verified_count += 1

        # Compute EVS trend
        evs_trend = self._compute_trend()

        # Apply adaptation rules based on phase and EVS
        adaptations = self._adapt(evs_snapshot, evs_trend)

        # Run SSGF outer step with adapted config
        ssgf_state = self.ssgf.outer_step()

        # Get audio mapping
        audio_params = self.ssgf.get_audio_mapping()

        # Check theurgic mode
        if audio_params.get("theurgic_mode", False):
            self._theurgic_count += 1

        snapshot = AdaptiveSnapshot(
            tick=self._tick,
            session_phase=self.session_phase.value,
            evs_score=evs_snapshot.evs_score,
            evs_trend=evs_trend,
            R_global=ssgf_state.R_global,
            audio_params=audio_params,
            ssgf_state=ssgf_state.to_dict(),
            adaptations_applied=adaptations,
        )
        self._snapshots.append(snapshot)
        self._tick += 1
        return snapshot

    def _compute_trend(self) -> float:
        """Compute EVS trend (positive = improving)."""
        window = self.config.evs_trend_window
        if len(self._evs_history) < 3:
            return 0.0
        recent = list(self._evs_history)[-window:]
        if len(recent) < 3:
            return 0.0
        # Simple linear regression slope
        x = np.arange(len(recent), dtype=float)
        y = np.array(recent)
        x_mean = x.mean()
        y_mean = y.mean()
        slope = float(np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-8))
        return slope

    def _adapt(self, evs: EVSSnapshot, trend: float) -> List[str]:
        """
        Apply adaptation rules based on session phase and EVS feedback.

        Returns list of adaptations applied.
        """
        adaptations = []
        rate = self.config.adaptation_rate
        phase = self.session_phase

        if phase == SessionPhase.DISCOVERY:
            # Phase 1: Gentle sweep, build baseline understanding
            if evs.evs_score < self.config.evs_low_threshold:
                # Low entrainment: broaden frequency, increase sigma_g
                self.ssgf.update_config(
                    sigma_g=min(0.6, self.ssgf.config.sigma_g + rate * 0.5),
                    noise_std=min(0.15, self.ssgf.config.noise_std + rate * 0.2),
                )
                adaptations.append("broaden_sweep")
            else:
                adaptations.append("discovery_ok")

        elif phase == SessionPhase.LOCK_ON:
            # Phase 2: Fine-tune toward responsive frequency
            if evs.evs_score < self.config.evs_low_threshold:
                # Still struggling: increase coupling strength
                self.ssgf.update_config(
                    sigma_g=min(0.8, self.ssgf.config.sigma_g + rate),
                    w_micro=min(2.0, self.ssgf.config.w_micro + rate * 0.3),
                )
                adaptations.append("increase_coupling")
            elif evs.evs_score >= self.config.evs_mid_threshold:
                # Good entrainment: tighten parameters
                self.ssgf.update_config(
                    lr_z=max(0.002, self.ssgf.config.lr_z * 0.95),
                    noise_std=max(0.01, self.ssgf.config.noise_std * 0.95),
                )
                adaptations.append("tighten_params")
            else:
                # Mid-range: gentle adjustment
                if trend < 0:
                    self.ssgf.update_config(
                        sigma_g=min(0.7, self.ssgf.config.sigma_g + rate * 0.2),
                    )
                    adaptations.append("boost_geometry")
                else:
                    adaptations.append("lock_on_stable")

        elif phase == SessionPhase.DEEPENING:
            # Phase 3: Deepen entrainment, work toward theurgic mode
            if evs.evs_score >= self.config.evs_mid_threshold:
                # Deepening: reduce learning rate for stability
                self.ssgf.update_config(
                    lr_z=max(0.001, self.ssgf.config.lr_z * 0.9),
                    field_pressure=min(0.3, self.ssgf.config.field_pressure + rate * 0.1),
                )
                adaptations.append("deepen_protocol")
            else:
                # Lost entrainment at depth: recover
                self.ssgf.update_config(
                    sigma_g=min(0.6, self.ssgf.config.sigma_g + rate * 0.3),
                )
                adaptations.append("recovery_boost")

        return adaptations

    def stop_session(self):
        """Stop the adaptive session."""
        self._running = False

    def get_session_report(self) -> AdaptiveSessionReport:
        """Generate post-session analytics report."""
        if not self._snapshots:
            return AdaptiveSessionReport()

        evs_scores = [s.evs_score for s in self._snapshots]
        total = len(self._snapshots)

        # Phase durations
        phase_counts = {}
        for s in self._snapshots:
            phase_counts[s.session_phase] = phase_counts.get(s.session_phase, 0) + 1

        # Compute grade
        verified_pct = (self._verified_count / total * 100) if total > 0 else 0
        if verified_pct >= 80:
            grade = "A"
        elif verified_pct >= 60:
            grade = "B"
        elif verified_pct >= 40:
            grade = "C"
        elif verified_pct >= 20:
            grade = "D"
        else:
            grade = "F"

        report = AdaptiveSessionReport(
            duration_s=float(self._tick),
            total_ticks=total,
            evs_avg=float(np.mean(evs_scores)),
            evs_peak=float(np.max(evs_scores)),
            evs_trend_final=self._compute_trend(),
            time_verified_pct=verified_pct,
            phase_durations={k: v for k, v in phase_counts.items()},
            theurgic_time_pct=(self._theurgic_count / total * 100) if total > 0 else 0,
            grade=grade,
        )

        # Update user profile
        self.profile.update_from_session(
            target_hz=self.evs.config.target_hz,
            evs_avg=report.evs_avg,
            evs_peak=report.evs_peak,
        )

        return report

    def get_state(self) -> Dict:
        """Get current engine state."""
        if self._snapshots:
            return self._snapshots[-1].to_dict()
        return AdaptiveSnapshot().to_dict()
