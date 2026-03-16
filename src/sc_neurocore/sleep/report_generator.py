# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Post-session sleep quality report generator

from __future__ import annotations

"""Post-session sleep quality report generator.

Analyses the tick history from a :class:`SleepOptimizer` session and
produces a composite quality score, letter grade, and actionable
recommendations.
"""


from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .sleep_stage_detector import SleepStage
from .sleep_optimizer import SleepOptimizer

# ---------------------------------------------------------------------------
# Report data
# ---------------------------------------------------------------------------


@dataclass
class SleepReport:
    """Aggregate report for a completed sleep session.

    Attributes
    ----------
    total_duration_min : float
        Total session length in minutes.
    sleep_onset_latency_min : float
        Minutes from session start until the first non-WAKE epoch.
    sleep_efficiency_pct : float
        Percentage of total time spent asleep (non-WAKE).
    quality_score : float
        Composite quality score in ``[0, 100]``.
    stage_durations_min : Dict[str, float]
        Time (minutes) per stage.
    stage_percentages : Dict[str, float]
        Percentage of total time per stage.
    stage_targets : Dict[str, float]
        Protocol target percentages for comparison.
    hypnogram : List[int]
        Stage codes per epoch.
    wakeups : int
        Number of WAKE epochs that occurred after initial sleep onset.
    reinductions : int
        Number of re-induction sequences triggered.
    recommendations : List[str]
        Plain-language suggestions.
    grade : str
        Letter grade (A-F).
    """

    total_duration_min: float = 0.0
    sleep_onset_latency_min: float = 0.0
    sleep_efficiency_pct: float = 0.0
    quality_score: float = 0.0
    stage_durations_min: Dict[str, float] = field(default_factory=dict)
    stage_percentages: Dict[str, float] = field(default_factory=dict)
    stage_targets: Dict[str, float] = field(default_factory=dict)
    hypnogram: List[int] = field(default_factory=list)
    wakeups: int = 0
    reinductions: int = 0
    recommendations: List[str] = field(default_factory=list)
    grade: str = "F"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class SleepReportGenerator:
    """Generates a :class:`SleepReport` from a completed optimiser session."""

    @staticmethod
    def generate(optimizer: SleepOptimizer) -> SleepReport:
        """Analyse *optimizer*'s tick history and return a report."""
        history = optimizer.get_history()
        if not history:
            return SleepReport()

        config = optimizer.config
        interval_min = config.stage_check_interval / (config.sample_rate * 60.0)

        # --- basic metrics ---------------------------------------------------
        total_min = len(history) * interval_min
        hypnogram = optimizer.get_hypnogram()

        # sleep onset latency
        sol_min = 0.0
        for tick in history:
            if tick.current_stage != SleepStage.WAKE:
                break
            sol_min += interval_min

        # stage durations
        durations = optimizer.get_stage_durations()
        dur_named: Dict[str, float] = {s.name: v for s, v in durations.items()}

        # stage percentages
        pct_named: Dict[str, float] = {}
        for s, v in durations.items():
            pct_named[s.name] = (v / total_min * 100.0) if total_min > 0 else 0.0

        # target percentages from protocol
        target_named: Dict[str, float] = {
            s.name: v * 100.0 for s, v in optimizer.protocol.stage_targets.items()
        }

        # sleep efficiency
        wake_min = durations.get(SleepStage.WAKE, 0.0)
        sleep_min = total_min - wake_min
        efficiency = (sleep_min / total_min * 100.0) if total_min > 0 else 0.0

        # wakeups after sleep onset
        sleep_started = False
        wakeups = 0
        in_wake = False
        for tick in history:
            if not sleep_started:
                if tick.current_stage != SleepStage.WAKE:
                    sleep_started = True
                    in_wake = False
                continue
            if tick.current_stage == SleepStage.WAKE:
                if not in_wake:
                    wakeups += 1
                    in_wake = True
            else:
                in_wake = False

        reinductions = optimizer._reinduction_count

        # --- quality score (0-100) -------------------------------------------
        # Component 1: stage match (40%)
        match_count = sum(1 for t in history if t.stage_match)
        stage_match_score = (match_count / len(history)) * 100.0 if history else 0.0

        # Component 2: sleep efficiency (25%)
        efficiency_score = min(100.0, efficiency / 0.85 * 100.0)  # 85% = perfect

        # Component 3: sleep onset latency (20%) -- <15 min is ideal
        if sol_min <= 15.0:
            sol_score = 100.0
        elif sol_min <= 30.0:
            sol_score = 100.0 - (sol_min - 15.0) / 15.0 * 50.0
        else:
            sol_score = max(0.0, 50.0 - (sol_min - 30.0) / 30.0 * 50.0)

        # Component 4: wakeups (15%) -- 0 = perfect, >= 5 = 0
        wakeup_score = max(0.0, 100.0 - wakeups * 20.0)

        quality = (
            stage_match_score * 0.40
            + efficiency_score * 0.25
            + sol_score * 0.20
            + wakeup_score * 0.15
        )
        quality = float(np.clip(quality, 0.0, 100.0))

        # --- grade -----------------------------------------------------------
        if quality >= 90:
            grade = "A"
        elif quality >= 75:
            grade = "B"
        elif quality >= 60:
            grade = "C"
        elif quality >= 40:
            grade = "D"
        else:
            grade = "F"

        # --- recommendations -------------------------------------------------
        recs: List[str] = []

        n3_pct = pct_named.get("N3", 0.0)
        n3_target = target_named.get("N3", 0.0)
        if n3_pct < n3_target * 0.7:
            recs.append(
                f"Deep sleep (N3) was {n3_pct:.1f}% vs target {n3_target:.1f}%. "
                "Consider the deep_sleep_boost protocol or earlier bedtime."
            )

        rem_pct = pct_named.get("REM", 0.0)
        rem_target = target_named.get("REM", 0.0)
        if rem_pct < rem_target * 0.7:
            recs.append(
                f"REM sleep was {rem_pct:.1f}% vs target {rem_target:.1f}%. "
                "Try the rem_enhancement protocol or extend sleep duration."
            )

        if sol_min > 20.0:
            recs.append(
                f"Sleep onset took {sol_min:.1f} min. "
                "The insomnia_relief protocol may help reduce latency."
            )

        if wakeups > 2:
            recs.append(
                f"You had {wakeups} awakenings. "
                "Reduce caffeine/alcohol and ensure a dark, cool environment."
            )

        if not recs:
            recs.append("Excellent session! Keep your current routine.")

        # --- assemble --------------------------------------------------------
        return SleepReport(
            total_duration_min=round(total_min, 2),
            sleep_onset_latency_min=round(sol_min, 2),
            sleep_efficiency_pct=round(efficiency, 2),
            quality_score=round(quality, 2),
            stage_durations_min={k: round(v, 2) for k, v in dur_named.items()},
            stage_percentages={k: round(v, 2) for k, v in pct_named.items()},
            stage_targets={k: round(v, 2) for k, v in target_named.items()},
            hypnogram=hypnogram,
            wakeups=wakeups,
            reinductions=reinductions,
            recommendations=recs,
            grade=grade,
        )
