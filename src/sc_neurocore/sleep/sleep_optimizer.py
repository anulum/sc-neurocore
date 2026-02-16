"""
Sleep Optimizer — Master closed-loop orchestrator
===================================================

Coordinates:
1. Circadian profiling → protocol selection
2. Real-time sleep stage detection from EEG
3. Adaptive audio parameter generation per stage
4. Closed-loop feedback: if wrong stage, adjust audio

Author: Claude (Session 2026-02-16)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .sleep_stage_detector import SleepStageDetector, SleepStage, DetectorConfig
from .circadian_optimizer import CircadianOptimizer, Chronotype
from .protocol_library import SleepProtocol, StageAudioParams, get_protocol


@dataclass
class SleepOptimizerConfig:
    """Sleep optimizer configuration."""
    sample_rate: int = 256
    fft_window: int = 512
    stage_check_interval: int = 256  # samples between stage checks
    max_reinduction_attempts: int = 3
    reinduction_cooldown_s: float = 60.0


@dataclass
class SleepTick:
    """Per-tick snapshot of the sleep optimizer."""
    tick: int = 0
    elapsed_min: float = 0.0
    current_stage: str = "WAKE"
    target_stage: str = "WAKE"
    stage_match: bool = True
    audio_params: Dict = field(default_factory=dict)
    band_powers: Dict[str, float] = field(default_factory=dict)
    reinduction_active: bool = False

    def to_dict(self) -> Dict:
        return {
            "tick": self.tick,
            "elapsed_min": round(self.elapsed_min, 2),
            "current_stage": self.current_stage,
            "target_stage": self.target_stage,
            "stage_match": self.stage_match,
            "audio_params": self.audio_params,
            "band_powers": {k: round(v, 4) for k, v in self.band_powers.items()},
            "reinduction_active": self.reinduction_active,
        }


class SleepOptimizer:
    """
    Master closed-loop sleep optimization engine.

    Usage:
        optimizer = SleepOptimizer(chronotype=Chronotype.BEAR)
        optimizer.start_session("insomnia_relief")
        while running:
            optimizer.add_sample(eeg_voltage)
            tick = optimizer.check_and_adapt()
        history = optimizer.get_history()
    """

    def __init__(
        self,
        chronotype: Chronotype = Chronotype.BEAR,
        config: Optional[SleepOptimizerConfig] = None,
    ):
        self.config = config or SleepOptimizerConfig()
        self.circadian = CircadianOptimizer(chronotype)
        self.detector = SleepStageDetector(DetectorConfig(
            sample_rate=self.config.sample_rate,
            fft_window=self.config.fft_window,
        ))

        self.protocol: Optional[SleepProtocol] = None
        self._tick = 0
        self._samples_since_check = 0
        self._start_time = 0.0
        self._running = False
        self._history: List[SleepTick] = []
        self._stage_durations: Dict[str, float] = {s.name: 0.0 for s in SleepStage}
        self._reinduction_count = 0
        self._last_reinduction_tick = -9999
        self._current_audio: Optional[StageAudioParams] = None

    def start_session(self, protocol_name: Optional[str] = None):
        """Start a sleep optimization session."""
        if protocol_name is None:
            protocol_name = self.circadian.get_recommended_protocol()
        self.protocol = get_protocol(protocol_name)
        self._tick = 0
        self._start_time = time.monotonic()
        self._running = True
        self._history.clear()
        self._stage_durations = {s.name: 0.0 for s in SleepStage}
        self._reinduction_count = 0
        self.detector.reset()

    def add_sample(self, voltage: float):
        """Add a single EEG voltage sample."""
        self.detector.add_sample(voltage)
        self._samples_since_check += 1

    def add_samples(self, voltages: np.ndarray):
        """Add multiple samples."""
        self.detector.add_samples(voltages)
        self._samples_since_check += len(voltages)

    def check_and_adapt(self) -> Optional[SleepTick]:
        """
        Check sleep stage and adapt audio if enough samples accumulated.

        Returns SleepTick if a check was performed, None otherwise.
        """
        if self._samples_since_check < self.config.stage_check_interval:
            return None

        self._samples_since_check = 0

        # Detect current stage
        current = self.detector.detect()
        elapsed_min = self._tick * self.config.stage_check_interval / self.config.sample_rate / 60.0
        total_min = self.protocol.duration_h * 60 if self.protocol else 480.0

        # Get target stage
        target = self.protocol.get_target_stage(elapsed_min, total_min) if self.protocol else SleepStage.N2

        # Track stage durations
        self._stage_durations[current.name] += self.config.stage_check_interval / self.config.sample_rate

        # Get audio parameters
        audio = self.protocol.get_audio_for_stage(current, elapsed_min) if self.protocol else StageAudioParams()
        self._current_audio = audio

        # Check for reinduction need
        reinduction = False
        if current == SleepStage.WAKE and target != SleepStage.WAKE:
            if (self.protocol and self.protocol.wake_recovery_enabled
                    and self._reinduction_count < self.config.max_reinduction_attempts
                    and (self._tick - self._last_reinduction_tick) > 10):
                reinduction = True
                self._reinduction_count += 1
                self._last_reinduction_tick = self._tick
                # Override audio with induction sweep
                sweep_hz = self.protocol.induction_sweep_start_hz
                audio = StageAudioParams(
                    binaural_hz=sweep_hz,
                    noise_color="brown",
                    volume=0.5,
                )

        tick = SleepTick(
            tick=self._tick,
            elapsed_min=elapsed_min,
            current_stage=current.name,
            target_stage=target.name,
            stage_match=current == target,
            audio_params={
                "binaural_hz": audio.binaural_hz,
                "noise_color": audio.noise_color,
                "volume": audio.volume,
                "base_freq_hz": audio.base_freq_hz,
                "isochronic_hz": audio.isochronic_hz,
            },
            band_powers=self.detector.get_band_powers(),
            reinduction_active=reinduction,
        )
        self._history.append(tick)
        self._tick += 1
        return tick

    def stop_session(self):
        """Stop the sleep session."""
        self._running = False

    def get_history(self) -> List[Dict]:
        """Return session history as list of dicts."""
        return [t.to_dict() for t in self._history]

    def get_stage_durations(self) -> Dict[str, float]:
        """Return time spent in each stage (seconds)."""
        return dict(self._stage_durations)

    def get_hypnogram(self) -> List[Dict]:
        """Return hypnogram data (time → stage transitions)."""
        if not self._history:
            return []
        result = []
        prev_stage = None
        for tick in self._history:
            if tick.current_stage != prev_stage:
                result.append({
                    "elapsed_min": tick.elapsed_min,
                    "stage": tick.current_stage,
                })
                prev_stage = tick.current_stage
        return result

    def get_state(self) -> Dict:
        """Current optimizer state."""
        if self._history:
            return self._history[-1].to_dict()
        return SleepTick().to_dict()
