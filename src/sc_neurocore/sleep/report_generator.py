"""
Sleep Report Generator — Morning report with hypnogram + metrics
=================================================================

Generates post-session analytics:
- Hypnogram visualization data
- Sleep quality composite score (0-100)
- Stage duration breakdown vs targets
- Recommendations for next session

Author: Claude (Session 2026-02-16)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .sleep_stage_detector import SleepStage
from .protocol_library import SleepProtocol
from .sleep_optimizer import SleepOptimizer


@dataclass
class SleepReport:
    """Morning sleep report."""
    total_duration_min: float = 0.0
    sleep_onset_latency_min: float = 0.0
    sleep_efficiency_pct: float = 0.0
    quality_score: float = 0.0  # 0-100
    stage_durations_min: Dict[str, float] = field(default_factory=dict)
    stage_percentages: Dict[str, float] = field(default_factory=dict)
    stage_targets: Dict[str, float] = field(default_factory=dict)
    hypnogram: List[Dict] = field(default_factory=list)
    wakeups: int = 0
    reinductions: int = 0
    recommendations: List[str] = field(default_factory=list)
    grade: str = "F"

    def to_dict(self) -> Dict:
        return {
            "total_duration_min": round(self.total_duration_min, 1),
            "sleep_onset_latency_min": round(self.sleep_onset_latency_min, 1),
            "sleep_efficiency_pct": round(self.sleep_efficiency_pct, 1),
            "quality_score": round(self.quality_score, 1),
            "stage_durations_min": {k: round(v, 1) for k, v in self.stage_durations_min.items()},
            "stage_percentages": {k: round(v, 1) for k, v in self.stage_percentages.items()},
            "stage_targets": self.stage_targets,
            "hypnogram": self.hypnogram,
            "wakeups": self.wakeups,
            "reinductions": self.reinductions,
            "recommendations": self.recommendations,
            "grade": self.grade,
        }


class SleepReportGenerator:
    """Generates morning report from completed sleep session."""

    def generate(self, optimizer: SleepOptimizer) -> SleepReport:
        """Generate report from a completed sleep optimizer session."""
        history = optimizer.get_history()
        if not history:
            return SleepReport()

        stage_durations_s = optimizer.get_stage_durations()
        stage_durations_min = {k: v / 60.0 for k, v in stage_durations_s.items()}
        total_min = sum(stage_durations_min.values())

        if total_min == 0:
            return SleepReport()

        # Stage percentages
        stage_pct = {k: (v / total_min * 100) for k, v in stage_durations_min.items()}

        # Sleep onset latency (time until first non-WAKE stage)
        sol_min = 0.0
        for tick in history:
            if tick["current_stage"] != "WAKE":
                break
            sol_min = tick["elapsed_min"]

        # Sleep efficiency: (total - WAKE) / total * 100
        wake_min = stage_durations_min.get("WAKE", 0)
        sleep_efficiency = ((total_min - wake_min) / total_min * 100) if total_min > 0 else 0

        # Count wakeups (transitions back to WAKE after sleep started)
        wakeups = 0
        was_asleep = False
        for tick in history:
            if tick["current_stage"] != "WAKE":
                was_asleep = True
            elif was_asleep and tick["current_stage"] == "WAKE":
                wakeups += 1
                was_asleep = False

        # Reinduction count
        reinductions = sum(1 for t in history if t.get("reinduction_active", False))

        # Stage targets from protocol
        stage_targets = {}
        if optimizer.protocol:
            stage_targets = {k: v * 100 for k, v in optimizer.protocol.stage_targets.items()}

        # Quality score (0-100)
        quality = self._compute_quality(
            stage_pct, stage_targets, sleep_efficiency, sol_min, wakeups
        )

        # Grade
        if quality >= 85:
            grade = "A"
        elif quality >= 70:
            grade = "B"
        elif quality >= 55:
            grade = "C"
        elif quality >= 40:
            grade = "D"
        else:
            grade = "F"

        # Recommendations
        recommendations = self._generate_recommendations(
            stage_pct, stage_targets, sol_min, wakeups, quality
        )

        return SleepReport(
            total_duration_min=total_min,
            sleep_onset_latency_min=sol_min,
            sleep_efficiency_pct=sleep_efficiency,
            quality_score=quality,
            stage_durations_min=stage_durations_min,
            stage_percentages=stage_pct,
            stage_targets=stage_targets,
            hypnogram=optimizer.get_hypnogram(),
            wakeups=wakeups,
            reinductions=reinductions,
            recommendations=recommendations,
            grade=grade,
        )

    def _compute_quality(
        self,
        actual_pct: Dict[str, float],
        target_pct: Dict[str, float],
        efficiency: float,
        sol_min: float,
        wakeups: int,
    ) -> float:
        """Compute sleep quality composite score."""
        score = 0.0

        # Component 1 (40%): Stage distribution match
        if target_pct:
            deviation = 0.0
            for stage in target_pct:
                actual = actual_pct.get(stage, 0)
                target = target_pct[stage]
                deviation += abs(actual - target) / max(target, 1)
            stage_match = max(0, 1.0 - deviation / len(target_pct))
            score += 40 * stage_match

        # Component 2 (25%): Sleep efficiency
        score += 25 * min(efficiency / 100, 1.0)

        # Component 3 (20%): Sleep onset latency (ideal < 15 min)
        if sol_min <= 15:
            score += 20
        elif sol_min <= 30:
            score += 20 * (1.0 - (sol_min - 15) / 15)
        # else 0

        # Component 4 (15%): Low wake-ups
        if wakeups == 0:
            score += 15
        elif wakeups <= 2:
            score += 15 * (1.0 - wakeups / 3)

        return float(np.clip(score, 0, 100))

    def _generate_recommendations(
        self,
        actual_pct: Dict[str, float],
        target_pct: Dict[str, float],
        sol_min: float,
        wakeups: int,
        quality: float,
    ) -> List[str]:
        """Generate actionable recommendations."""
        recs = []

        if sol_min > 30:
            recs.append("Sleep onset was slow. Try the insomnia_relief protocol with earlier bedtime.")

        n3_actual = actual_pct.get("N3", 0)
        n3_target = target_pct.get("N3", 20)
        if n3_actual < n3_target * 0.6:
            recs.append("Deep sleep (N3) was below target. Consider the deep_sleep_boost protocol.")

        rem_actual = actual_pct.get("REM", 0)
        rem_target = target_pct.get("REM", 20)
        if rem_actual < rem_target * 0.6:
            recs.append("REM sleep was below target. Try rem_enhancement protocol.")

        if wakeups > 3:
            recs.append("Multiple wakeups detected. Reduce caffeine and screen time before bed.")

        wake_pct = actual_pct.get("WAKE", 0)
        if wake_pct > 15:
            recs.append("Too much time awake in bed. Consider stimulus control therapy.")

        if quality >= 80:
            recs.append("Excellent sleep quality! Continue current protocol.")

        if not recs:
            recs.append("Good sleep session. No major issues detected.")

        return recs
