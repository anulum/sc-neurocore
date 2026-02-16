"""
Sleep Protocol Library — Templates for different sleep goals
==============================================================

Each protocol defines per-stage audio parameters and transition rules.

Protocols:
    insomnia_relief   - Progressive delta sweep for difficulty falling asleep
    jet_lag_reset     - Phase-shift audio for travelers
    deep_sleep_boost  - Maximize N3 time for recovery
    rem_enhancement   - Extend REM for creativity/memory
    shift_worker      - Compressed polyphasic support
    power_nap         - Quick 25-min N2 induction

Author: Claude (Session 2026-02-16)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .sleep_stage_detector import SleepStage


@dataclass
class StageAudioParams:
    """Audio parameters for a specific sleep stage."""
    binaural_hz: float = 10.0      # Binaural beat frequency
    noise_color: str = "pink"       # pink, brown, white
    base_freq_hz: float = 200.0    # Carrier frequency
    volume: float = 0.5            # 0-1
    isochronic_hz: float = 0.0     # 0 = disabled
    spatial_rotation: bool = False


@dataclass
class SleepProtocol:
    """Complete sleep protocol definition."""
    name: str
    description: str
    target_audience: str
    duration_h: float
    # Per-stage audio parameters
    stage_params: Dict[str, StageAudioParams] = field(default_factory=dict)
    # Stage duration targets (fraction of total)
    stage_targets: Dict[str, float] = field(default_factory=dict)
    # Transition rules
    induction_sweep_start_hz: float = 10.0  # Start frequency for induction
    induction_sweep_end_hz: float = 2.0     # End frequency
    induction_duration_min: float = 30.0
    wake_recovery_enabled: bool = True  # Re-induction if wake detected

    def get_audio_for_stage(
        self, current_stage: SleepStage, elapsed_min: float = 0.0
    ) -> StageAudioParams:
        """Get audio parameters for the current stage and time."""
        stage_key = current_stage.name
        if stage_key in self.stage_params:
            return self.stage_params[stage_key]
        # Default fallback
        return StageAudioParams(binaural_hz=4.0, noise_color="pink", volume=0.3)

    def get_target_stage(self, elapsed_min: float, total_min: float) -> SleepStage:
        """Get the target sleep stage for the current time in the session."""
        progress = elapsed_min / max(total_min, 1.0)
        if progress < 0.05:
            return SleepStage.WAKE  # Initial settling
        elif progress < 0.15:
            return SleepStage.N1
        elif progress < 0.25:
            return SleepStage.N2
        elif progress < 0.50:
            return SleepStage.N3
        elif progress < 0.70:
            return SleepStage.N2
        elif progress < 0.85:
            return SleepStage.REM
        else:
            return SleepStage.N1  # Approaching wake

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "target_audience": self.target_audience,
            "duration_h": self.duration_h,
            "induction_sweep": f"{self.induction_sweep_start_hz}→{self.induction_sweep_end_hz} Hz",
            "stages": list(self.stage_params.keys()),
        }


# ── Protocol Definitions ─────────────────────────────────────────────────

INSOMNIA_RELIEF = SleepProtocol(
    name="insomnia_relief",
    description="Progressive delta sweep 10Hz→2Hz for difficulty falling asleep",
    target_audience="Difficulty falling asleep",
    duration_h=8.0,
    induction_sweep_start_hz=10.0,
    induction_sweep_end_hz=2.0,
    induction_duration_min=30.0,
    stage_params={
        "WAKE": StageAudioParams(binaural_hz=10.0, noise_color="pink", volume=0.4),
        "N1": StageAudioParams(binaural_hz=6.0, noise_color="pink", volume=0.35),
        "N2": StageAudioParams(binaural_hz=4.0, noise_color="brown", volume=0.3),
        "N3": StageAudioParams(binaural_hz=1.5, noise_color="brown", volume=0.25),
        "REM": StageAudioParams(binaural_hz=5.0, noise_color="pink", volume=0.2),
    },
    stage_targets={"WAKE": 0.02, "N1": 0.08, "N2": 0.45, "N3": 0.25, "REM": 0.20},
)

JET_LAG_RESET = SleepProtocol(
    name="jet_lag_reset",
    description="Phase-shift audio with timed melatonin-window and morning alpha",
    target_audience="Travelers crossing time zones",
    duration_h=7.0,
    induction_sweep_start_hz=8.0,
    induction_sweep_end_hz=1.5,
    induction_duration_min=20.0,
    stage_params={
        "WAKE": StageAudioParams(binaural_hz=12.0, noise_color="white", volume=0.3),
        "N1": StageAudioParams(binaural_hz=7.0, noise_color="pink", volume=0.3),
        "N2": StageAudioParams(binaural_hz=4.0, noise_color="pink", volume=0.3),
        "N3": StageAudioParams(binaural_hz=1.0, noise_color="brown", volume=0.25),
        "REM": StageAudioParams(binaural_hz=6.0, noise_color="pink", volume=0.2),
    },
    stage_targets={"WAKE": 0.05, "N1": 0.10, "N2": 0.40, "N3": 0.25, "REM": 0.20},
)

DEEP_SLEEP_BOOST = SleepProtocol(
    name="deep_sleep_boost",
    description="Maximize N3 time at 0.75Hz binaural for recovery",
    target_audience="Athletes, physical recovery",
    duration_h=8.0,
    induction_sweep_start_hz=8.0,
    induction_sweep_end_hz=0.75,
    induction_duration_min=25.0,
    stage_params={
        "WAKE": StageAudioParams(binaural_hz=8.0, noise_color="pink", volume=0.4),
        "N1": StageAudioParams(binaural_hz=4.0, noise_color="pink", volume=0.35),
        "N2": StageAudioParams(binaural_hz=2.0, noise_color="brown", volume=0.3),
        "N3": StageAudioParams(binaural_hz=0.75, noise_color="brown", volume=0.3, isochronic_hz=0.5),
        "REM": StageAudioParams(binaural_hz=5.0, noise_color="pink", volume=0.2),
    },
    stage_targets={"WAKE": 0.02, "N1": 0.05, "N2": 0.35, "N3": 0.38, "REM": 0.20},
)

REM_ENHANCEMENT = SleepProtocol(
    name="rem_enhancement",
    description="Extended theta windows for creativity and memory consolidation",
    target_audience="Creativity, memory improvement",
    duration_h=8.0,
    induction_sweep_start_hz=10.0,
    induction_sweep_end_hz=3.0,
    induction_duration_min=30.0,
    stage_params={
        "WAKE": StageAudioParams(binaural_hz=10.0, noise_color="pink", volume=0.35),
        "N1": StageAudioParams(binaural_hz=7.0, noise_color="pink", volume=0.3),
        "N2": StageAudioParams(binaural_hz=4.0, noise_color="pink", volume=0.3),
        "N3": StageAudioParams(binaural_hz=1.5, noise_color="brown", volume=0.25),
        "REM": StageAudioParams(binaural_hz=6.0, noise_color="pink", volume=0.3, spatial_rotation=True),
    },
    stage_targets={"WAKE": 0.02, "N1": 0.08, "N2": 0.35, "N3": 0.20, "REM": 0.35},
)

SHIFT_WORKER = SleepProtocol(
    name="shift_worker",
    description="Compressed polyphasic, aggressive induction for night shift",
    target_audience="Night shift workers",
    duration_h=5.0,
    induction_sweep_start_hz=8.0,
    induction_sweep_end_hz=1.0,
    induction_duration_min=15.0,
    stage_params={
        "WAKE": StageAudioParams(binaural_hz=8.0, noise_color="brown", volume=0.5),
        "N1": StageAudioParams(binaural_hz=5.0, noise_color="brown", volume=0.4),
        "N2": StageAudioParams(binaural_hz=3.0, noise_color="brown", volume=0.35),
        "N3": StageAudioParams(binaural_hz=0.75, noise_color="brown", volume=0.3),
        "REM": StageAudioParams(binaural_hz=5.0, noise_color="pink", volume=0.25),
    },
    stage_targets={"WAKE": 0.02, "N1": 0.05, "N2": 0.40, "N3": 0.30, "REM": 0.23},
)

POWER_NAP = SleepProtocol(
    name="power_nap",
    description="Quick 25min N2 induction with alarm",
    target_audience="Afternoon reset",
    duration_h=25.0 / 60.0,
    induction_sweep_start_hz=10.0,
    induction_sweep_end_hz=3.0,
    induction_duration_min=10.0,
    stage_params={
        "WAKE": StageAudioParams(binaural_hz=10.0, noise_color="pink", volume=0.3),
        "N1": StageAudioParams(binaural_hz=6.0, noise_color="pink", volume=0.3),
        "N2": StageAudioParams(binaural_hz=3.0, noise_color="pink", volume=0.3),
        "N3": StageAudioParams(binaural_hz=3.0, noise_color="pink", volume=0.2),  # Avoid deep
        "REM": StageAudioParams(binaural_hz=6.0, noise_color="pink", volume=0.2),
    },
    stage_targets={"WAKE": 0.10, "N1": 0.30, "N2": 0.50, "N3": 0.05, "REM": 0.05},
    wake_recovery_enabled=False,
)

PROTOCOL_REGISTRY: Dict[str, SleepProtocol] = {
    "insomnia_relief": INSOMNIA_RELIEF,
    "jet_lag_reset": JET_LAG_RESET,
    "deep_sleep_boost": DEEP_SLEEP_BOOST,
    "rem_enhancement": REM_ENHANCEMENT,
    "shift_worker": SHIFT_WORKER,
    "power_nap": POWER_NAP,
}


def get_protocol(name: str) -> SleepProtocol:
    """Get a protocol by name."""
    if name not in PROTOCOL_REGISTRY:
        raise ValueError(f"Unknown protocol: {name}. Available: {list(PROTOCOL_REGISTRY.keys())}")
    return PROTOCOL_REGISTRY[name]


def list_protocols() -> List[Dict]:
    """List all available protocols."""
    return [p.to_dict() for p in PROTOCOL_REGISTRY.values()]
