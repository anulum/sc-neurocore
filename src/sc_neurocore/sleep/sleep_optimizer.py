from __future__ import annotations
from typing import Any, Optional

"""Closed-loop sleep optimisation engine.

Combines the sleep-stage detector, a selected protocol, and an adaptive
audio pipeline into a tick-by-tick optimiser that tracks stage progression,
detects unwanted awakenings, and triggers re-induction sequences.
"""


from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .sleep_stage_detector import DetectorConfig, SleepStage, SleepStageDetector
from .protocol_library import SleepProtocol, StageAudioParams, get_protocol


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SleepOptimizerConfig:
    """Tuneable knobs for the optimiser loop."""

    sample_rate: int = 256
    fft_window: int = 512
    stage_check_interval: int = 256
    max_reinduction_attempts: int = 3


# ---------------------------------------------------------------------------
# Per-tick output
# ---------------------------------------------------------------------------


@dataclass
class SleepTick:
    """Snapshot produced every ``stage_check_interval`` samples.

    Attributes
    ----------
    tick : int
        Monotonic tick counter.
    elapsed_min : float
        Wall-clock minutes since session start.
    current_stage : SleepStage
        Detected stage at this tick.
    target_stage : SleepStage
        Protocol's ideal stage at this progress point.
    stage_match : bool
        Whether current == target.
    audio_params : StageAudioParams
        Audio parameters being delivered.
    band_powers : Dict[str, float]
        Most recent EEG band-power decomposition.
    reinduction_active : bool
        Whether a re-induction sequence is currently running.
    """

    tick: int = 0
    elapsed_min: float = 0.0
    current_stage: SleepStage = SleepStage.WAKE
    target_stage: SleepStage = SleepStage.WAKE
    stage_match: bool = True
    audio_params: StageAudioParams = field(default_factory=StageAudioParams)
    band_powers: Dict[str, float] = field(default_factory=dict)
    reinduction_active: bool = False


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class SleepOptimizer:
    """Closed-loop sleep optimiser.

    Parameters
    ----------
    protocol : SleepProtocol or str
        The protocol instance (or its registry name) to follow.
    config : SleepOptimizerConfig, optional
        Operational parameters.

    Example::

        opt = SleepOptimizer("insomnia_relief")
        opt.start_session()
        for sample in eeg_stream:
            opt.add_sample(sample)
            tick = opt.check_and_adapt()
            if tick is not None:
                apply_audio(tick.audio_params)
        report = opt.stop_session()
    """

    def __init__(
        self,
        protocol: SleepProtocol | str,
        config: Optional[SleepOptimizerConfig] = None,
    ) -> None:
        if isinstance(protocol, str):
            protocol = get_protocol(protocol)
        self.protocol: SleepProtocol = protocol
        self.config = config or SleepOptimizerConfig()

        det_cfg = DetectorConfig(
            sample_rate=self.config.sample_rate,
            fft_window=self.config.fft_window,
        )
        self._detector = SleepStageDetector(det_cfg)

        # session state
        self._active: bool = False
        self._sample_count: int = 0
        self._tick_count: int = 0
        self._history: List[SleepTick] = []
        self._reinduction_count: int = 0
        self._reinduction_active: bool = False
        self._consecutive_wake: int = 0

    # -- session lifecycle --------------------------------------------------

    def start_session(self) -> None:
        """Begin a new optimisation session, resetting all state."""
        self._detector.reset()
        self._active = True
        self._sample_count = 0
        self._tick_count = 0
        self._history = []
        self._reinduction_count = 0
        self._reinduction_active = False
        self._consecutive_wake = 0

    def stop_session(self) -> List[SleepTick]:
        """End the current session and return the full tick history."""
        self._active = False
        return list(self._history)

    # -- sample ingestion ---------------------------------------------------

    def add_sample(self, sample: float) -> None:
        """Feed a single EEG voltage sample."""
        if not self._active:
            return
        self._detector.add_sample(sample)
        self._sample_count += 1

    def add_samples(self, samples: np.ndarray[Any, Any]) -> None:
        """Feed an array of EEG voltage samples."""
        if not self._active:
            return
        self._detector.add_samples(samples)
        self._sample_count += len(np.asarray(samples).ravel())

    # -- adaptation ---------------------------------------------------------

    def check_and_adapt(self) -> Optional[SleepTick]:
        """Run stage detection and protocol adaptation.

        Should be called after every ``stage_check_interval`` samples.
        Returns a :class:`SleepTick` when a check is performed, or
        ``None`` if the interval has not elapsed or the session is
        inactive.
        """
        if not self._active:
            return None
        if self._sample_count < (self._tick_count + 1) * self.config.stage_check_interval:
            return None

        self._tick_count += 1

        stage = self._detector.detect()
        if stage is None:
            stage = SleepStage.WAKE

        total_dur_samples = self.protocol.total_duration_min * 60.0 * self.config.sample_rate
        progress = (
            min(1.0, self._sample_count / total_dur_samples) if total_dur_samples > 0 else 0.0
        )
        target = self.protocol.get_target_stage(progress)

        # reinduction logic: detect unwanted awakenings
        if stage == SleepStage.WAKE and target != SleepStage.WAKE:
            self._consecutive_wake += 1
            if (
                self._consecutive_wake >= 2
                and self._reinduction_count < self.config.max_reinduction_attempts
            ):
                self._reinduction_active = True
                self._reinduction_count += 1
        else:
            self._consecutive_wake = 0
            self._reinduction_active = False

        # select audio: during reinduction use N1 params to gently re-induce
        if self._reinduction_active:
            audio = self.protocol.get_audio_for_stage(SleepStage.N1)
        else:
            audio = self.protocol.get_audio_for_stage(stage)

        elapsed_min = self._sample_count / (self.config.sample_rate * 60.0)
        band_powers = self._detector.get_band_powers() or {}

        tick = SleepTick(
            tick=self._tick_count,
            elapsed_min=elapsed_min,
            current_stage=stage,
            target_stage=target,
            stage_match=(stage == target),
            audio_params=audio,
            band_powers=band_powers,
            reinduction_active=self._reinduction_active,
        )
        self._history.append(tick)
        return tick

    # -- query --------------------------------------------------------------

    def get_history(self) -> List[SleepTick]:
        """Return a copy of all recorded ticks."""
        return list(self._history)

    def get_stage_durations(self) -> Dict[SleepStage, float]:
        """Compute time (minutes) spent in each detected stage."""
        interval_min = self.config.stage_check_interval / (self.config.sample_rate * 60.0)
        durations: Dict[SleepStage, float] = {s: 0.0 for s in SleepStage}
        for tick in self._history:
            durations[tick.current_stage] += interval_min
        return durations

    def get_hypnogram(self) -> List[int]:
        """Return the detected-stage sequence as a list of integer codes."""
        return [int(t.current_stage) for t in self._history]

    def get_state(self) -> Dict[str, Any]:
        """Return a summary dict of the optimiser's current state."""
        last = self._history[-1] if self._history else None
        return {
            "active": self._active,
            "tick_count": self._tick_count,
            "sample_count": self._sample_count,
            "elapsed_min": (
                self._sample_count / (self.config.sample_rate * 60.0) if self._active else 0.0
            ),
            "current_stage": last.current_stage.name if last else None,
            "target_stage": last.target_stage.name if last else None,
            "reinduction_count": self._reinduction_count,
            "reinduction_active": self._reinduction_active,
            "protocol": self.protocol.name,
        }
