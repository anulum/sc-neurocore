# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive Audio Engine -- Closed-Loop SSGF + EVS Controller

"""Closed-loop SSGF and EVS adaptive audio controller.

Orchestrates a three-phase adaptive audio session:

    DISCOVERY   (0-2 min)   -- gentle frequency sweep, find resonance
    LOCK_ON     (2-10 min)  -- lock on optimal frequency, responsive
    DEEPENING   (10+ min)   -- push toward theurgic coherence

Each tick receives an EVSSnapshot and returns adjusted audio parameters
by modulating the SSGFEngine configuration (sigma_g, lr_z, field_pressure).

"""

from __future__ import annotations


import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from .ssgf_engine import SSGFEngine
from .evs_engine import EVSEngine, EVSSnapshot
from .user_profile import UserProfile

logger = logging.getLogger(__name__)

# ── Session Phase ────────────────────────────────────────────────────


class SessionPhase(str, Enum):
    """Adaptive audio control phase for a closed-loop session."""

    DISCOVERY = "discovery"
    LOCK_ON = "lock_on"
    DEEPENING = "deepening"


# Phase transition thresholds (in ticks at ~2 Hz update rate)
_DISCOVERY_TICKS = 240  # ~2 minutes
_LOCKON_TICKS = 1200  # ~10 minutes


# ── Adaptation Record ────────────────────────────────────────────────


@dataclass
class _AdaptationRecord:
    """Single parameter adaptation event."""

    tick: int
    phase: str
    param: str
    old_value: float
    new_value: float
    reason: str


# ── Session Report ───────────────────────────────────────────────────


@dataclass
class AdaptiveSessionReport:
    """Summary of a completed adaptive audio session."""

    total_ticks: int = 0
    avg_evs: float = 0.0
    peak_evs: float = 0.0
    verified_pct: float = 0.0
    grade: str = "F"
    adaptations: int = 0
    phase_durations: dict[str, int] = field(default_factory=dict)
    final_audio: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible summary of the adaptive session."""
        return {
            "total_ticks": self.total_ticks,
            "avg_evs": round(self.avg_evs, 2),
            "peak_evs": round(self.peak_evs, 2),
            "verified_pct": round(self.verified_pct, 2),
            "grade": self.grade,
            "adaptations": self.adaptations,
            "phase_durations": self.phase_durations,
            "final_audio": self.final_audio,
        }


def _compute_grade(verified_pct: float) -> str:
    """Map verified percentage to letter grade."""
    if verified_pct >= 80.0:
        return "A"
    if verified_pct >= 60.0:
        return "B"
    if verified_pct >= 40.0:
        return "C"
    if verified_pct >= 20.0:
        return "D"
    return "F"


# ── Engine ───────────────────────────────────────────────────────────


class AdaptiveAudioEngine:
    """Closed-loop adaptive audio controller coupling SSGF with EVS.

    Parameters
    ----------
    ssgf : SSGFEngine
        The geometry solver producing audio mappings.
    evs : EVSEngine
        The entrainment verification scorer.
    profile : UserProfile, optional
        User preferences for chronotype-aware adaptation.
    """

    def __init__(
        self,
        ssgf: SSGFEngine,
        evs: EVSEngine,
        profile: UserProfile | None = None,
    ):
        self.ssgf = ssgf
        self.evs = evs
        self.profile = profile

        # Session state
        self._tick: int = 0
        self._phase: SessionPhase = SessionPhase.DISCOVERY
        self._phase_start_tick: int = 0

        # EVS tracking
        self._evs_scores: list[float] = []
        self._verified_count: int = 0

        # Trend detection
        self._recent_evs: list[float] = []
        self._trend_window: int = 10

        # Adaptation log
        self._adaptations: list[_AdaptationRecord] = []

        # Discovery sweep state
        self._sweep_direction: float = 1.0
        self._sweep_hz: float = 10.0 if profile is None else profile.get_best_target_hz()

    # ── Phase Management ─────────────────────────────────────────────

    def _update_phase(self) -> None:
        """Transition between session phases based on tick count."""
        if self._phase == SessionPhase.DISCOVERY and self._tick >= _DISCOVERY_TICKS:
            self._phase = SessionPhase.LOCK_ON
            self._phase_start_tick = self._tick
            logger.info("Session phase -> LOCK_ON at tick %d", self._tick)
        elif self._phase == SessionPhase.LOCK_ON and self._tick >= _LOCKON_TICKS:
            self._phase = SessionPhase.DEEPENING
            self._phase_start_tick = self._tick
            logger.info("Session phase -> DEEPENING at tick %d", self._tick)

    # ── Trend Analysis ───────────────────────────────────────────────

    def _evs_trend(self) -> float:
        """Return recent EVS trend: positive = improving, negative = declining."""
        if len(self._recent_evs) < 3:
            return 0.0
        window = max(self._trend_window, 3)
        recent = np.array(self._recent_evs[-window:])
        # Simple linear slope
        x = np.arange(len(recent), dtype=np.float64)
        x_mean = x.mean()
        y_mean = recent.mean()
        denom = np.sum((x - x_mean) ** 2)
        slope = np.sum((x - x_mean) * (recent - y_mean)) / denom
        return float(slope)

    # ── Core Tick ────────────────────────────────────────────────────

    def on_evs_update(self, snapshot: EVSSnapshot) -> dict[str, float]:
        """Process one EVS update and return adapted audio parameters.

        This is the main feedback loop entry point.  Call it each time
        a new EVSSnapshot is available (~every 500 ms).

        Returns
        -------
        dict
            Audio parameters from SSGF, possibly adjusted by adaptation.
        """
        self._tick += 1
        self._update_phase()

        # Track EVS
        score = snapshot.evs_score
        self._evs_scores.append(score)
        self._recent_evs.append(score)
        if len(self._recent_evs) > self._trend_window * 2:
            self._recent_evs = self._recent_evs[-self._trend_window * 2 :]
        if snapshot.is_verified:
            self._verified_count += 1

        trend = self._evs_trend()

        # Phase-specific adaptation
        if self._phase == SessionPhase.DISCOVERY:
            self._adapt_discovery(snapshot, trend)
        elif self._phase == SessionPhase.LOCK_ON:
            self._adapt_lock_on(snapshot, trend)
        else:
            self._adapt_deepening(snapshot, trend)

        # Run one SSGF outer step to update geometry
        self.ssgf.outer_step()

        return self.ssgf.get_audio_mapping()

    # ── Phase-Specific Adaptation ────────────────────────────────────

    def _adapt_discovery(self, snap: EVSSnapshot, trend: float) -> None:
        """Discovery phase: gentle frequency sweep, widen geometry."""
        cfg = self.ssgf.cfg

        # Sweep target Hz slowly
        self._sweep_hz += self._sweep_direction * 0.1
        if self._sweep_hz > 15.0:
            self._sweep_direction = -1.0
        elif self._sweep_hz < 5.0:
            self._sweep_direction = 1.0
        self.evs.set_target(self._sweep_hz)

        # Keep sigma_g moderate for exploration
        old_sg = cfg.sigma_g
        cfg.sigma_g = float(np.clip(cfg.sigma_g, 0.15, 0.35))
        if cfg.sigma_g != old_sg:
            self._log_adaptation("sigma_g", old_sg, cfg.sigma_g, "discovery bounds")

        # Higher learning rate for faster geometry search
        old_lr = cfg.lr_z
        cfg.lr_z = 0.015
        if cfg.lr_z != old_lr:
            self._log_adaptation("lr_z", old_lr, cfg.lr_z, "discovery exploration")

    def _adapt_lock_on(self, snap: EVSSnapshot, trend: float) -> None:
        """Lock-On phase: responsive frequency tracking, tighten geometry."""
        cfg = self.ssgf.cfg

        # If EVS is declining, increase geometry feedback
        if trend < -0.5:
            old_sg = cfg.sigma_g
            new_sg = float(np.clip(cfg.sigma_g + 0.02, 0.1, 0.6))
            if new_sg != old_sg:
                cfg.sigma_g = new_sg
                self._log_adaptation("sigma_g", old_sg, new_sg, "EVS declining, boost coupling")

        # If EVS is improving, reduce learning rate to stabilise
        if trend > 0.5:
            old_lr = cfg.lr_z
            new_lr = float(np.clip(cfg.lr_z * 0.95, 0.002, 0.02))
            if new_lr != old_lr:
                cfg.lr_z = new_lr
                self._log_adaptation("lr_z", old_lr, new_lr, "EVS improving, stabilise")

        # Responsive target adjustment based on peak alignment
        if snap.peak_alignment < 0.5 and snap.peak_hz > 0.5:
            # Nudge target toward actual brain peak
            delta = (snap.peak_hz - snap.target_hz) * 0.1
            new_target = float(np.clip(snap.target_hz + delta, 0.5, 40.0))
            self.evs.set_target(new_target)

    def _adapt_deepening(self, snap: EVSSnapshot, trend: float) -> None:
        """Deepening phase: push toward theurgic coherence."""
        cfg = self.ssgf.cfg

        # Increase field pressure to encourage synchrony
        old_fp = cfg.field_pressure
        pressure_cap = 0.5 if self.ssgf.R_global > 0.9 else 0.4
        new_fp = float(np.clip(cfg.field_pressure + 0.005, 0.05, pressure_cap))
        if new_fp != old_fp:
            cfg.field_pressure = new_fp
            self._log_adaptation("field_pressure", old_fp, new_fp, "deepening push")

        # Increase sigma_g gradually
        old_sg = cfg.sigma_g
        new_sg = float(np.clip(cfg.sigma_g + 0.005, 0.2, 0.8))
        if new_sg != old_sg:
            cfg.sigma_g = new_sg
            self._log_adaptation("sigma_g", old_sg, new_sg, "deepening geometry boost")

        # Lower learning rate for stability
        old_lr = cfg.lr_z
        new_lr = float(np.clip(cfg.lr_z * 0.98, 0.001, 0.01))
        if new_lr != old_lr:
            cfg.lr_z = new_lr
            self._log_adaptation("lr_z", old_lr, new_lr, "deepening stabilise")

        # If R > 0.9, we're close to theurgic -- fine-tune
        if self.ssgf.R_global > 0.9:
            old_fp2 = cfg.field_pressure
            new_fp2 = float(np.clip(cfg.field_pressure + 0.01, 0.1, 0.5))
            if new_fp2 != old_fp2:
                cfg.field_pressure = new_fp2
                self._log_adaptation("field_pressure", old_fp2, new_fp2, "near-theurgic push")

    # ── Logging ──────────────────────────────────────────────────────

    def _log_adaptation(
        self,
        param: str,
        old: float,
        new: float,
        reason: str,
    ) -> None:
        record = _AdaptationRecord(
            tick=self._tick,
            phase=self._phase.value,
            param=param,
            old_value=old,
            new_value=new,
            reason=reason,
        )
        self._adaptations.append(record)
        logger.debug(
            "Tick %d [%s] %s: %.4f -> %.4f (%s)",
            self._tick,
            self._phase.value,
            param,
            old,
            new,
            reason,
        )

    # ── Session Report ───────────────────────────────────────────────

    def get_session_report(self) -> AdaptiveSessionReport:
        """Generate summary report of the current session."""
        total = len(self._evs_scores)
        avg_evs = float(np.mean(self._evs_scores)) if self._evs_scores else 0.0
        peak_evs = float(np.max(self._evs_scores)) if self._evs_scores else 0.0
        verified_pct = (self._verified_count / total * 100.0) if total > 0 else 0.0

        # Phase durations
        phase_durations: dict[str, int] = {}
        if self._tick > 0:
            if self._tick <= _DISCOVERY_TICKS:
                phase_durations["discovery"] = self._tick
            elif self._tick <= _LOCKON_TICKS:
                phase_durations["discovery"] = _DISCOVERY_TICKS
                phase_durations["lock_on"] = self._tick - _DISCOVERY_TICKS
            else:
                phase_durations["discovery"] = _DISCOVERY_TICKS
                phase_durations["lock_on"] = _LOCKON_TICKS - _DISCOVERY_TICKS
                phase_durations["deepening"] = self._tick - _LOCKON_TICKS

        return AdaptiveSessionReport(
            total_ticks=total,
            avg_evs=avg_evs,
            peak_evs=peak_evs,
            verified_pct=verified_pct,
            grade=_compute_grade(verified_pct),
            adaptations=len(self._adaptations),
            phase_durations=phase_durations,
            final_audio=self.ssgf.get_audio_mapping(),
        )

    # ── Utilities ────────────────────────────────────────────────────

    @property
    def current_phase(self) -> SessionPhase:
        """Return the active adaptive-control phase."""
        return self._phase

    @property
    def tick(self) -> int:
        """Return the number of processed EVS updates."""
        return self._tick

    def reset(self) -> None:
        """Reset session state (does not reset SSGF or EVS)."""
        self._tick = 0
        self._phase = SessionPhase.DISCOVERY
        self._phase_start_tick = 0
        self._evs_scores.clear()
        self._verified_count = 0
        self._recent_evs.clear()
        self._adaptations.clear()
        self._sweep_direction = 1.0
        self._sweep_hz = 10.0 if self.profile is None else self.profile.get_best_target_hz()
