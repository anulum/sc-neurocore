# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
from typing import Any, Optional

"""Library of pre-built sleep audio protocols.

Each protocol maps every AASM sleep stage to a set of audio-entrainment
parameters (binaural beat frequency, noise colour, isochronic pulse rate,
etc.) and specifies ideal stage-time targets for the night.
"""


from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .sleep_stage_detector import SleepStage


# ---------------------------------------------------------------------------
# Audio parameters per stage
# ---------------------------------------------------------------------------


@dataclass
class StageAudioParams:
    """Audio-entrainment parameters for a single sleep stage.

    Attributes
    ----------
    binaural_hz : float
        Binaural beat frequency (Hz).
    noise_color : str
        Background noise colour (``"pink"``, ``"brown"``, ``"white"``, etc.).
    base_freq_hz : float
        Carrier / base tone frequency (Hz).
    volume : float
        Relative volume in ``[0, 1]``.
    isochronic_hz : float
        Isochronic pulse frequency (Hz); 0 disables.
    spatial_rotation : float
        Spatial audio rotation speed in degrees per second.
    """

    binaural_hz: float = 2.0
    noise_color: str = "pink"
    base_freq_hz: float = 200.0
    volume: float = 0.5
    isochronic_hz: float = 0.0
    spatial_rotation: float = 0.0


# ---------------------------------------------------------------------------
# Sleep protocol
# ---------------------------------------------------------------------------


@dataclass
class SleepProtocol:
    """A named sleep-entrainment protocol.

    Attributes
    ----------
    name : str
        Human-readable identifier (must match the registry key).
    description : str
        Short description of the protocol's therapeutic goal.
    stage_audio : Dict[SleepStage, StageAudioParams]
        Audio parameters keyed by target stage.
    stage_targets : Dict[SleepStage, float]
        Target fraction of total sleep time per stage (must sum to 1.0).
    total_duration_min : float
        Recommended session length in minutes.
    """

    name: str = ""
    description: str = ""
    stage_audio: Dict[SleepStage, StageAudioParams] = field(default_factory=dict)
    stage_targets: Dict[SleepStage, float] = field(default_factory=dict)
    total_duration_min: float = 480.0  # 8 hours default

    # -- public API ---------------------------------------------------------

    def get_audio_for_stage(self, stage: SleepStage) -> StageAudioParams:
        """Return audio parameters for *stage*, falling back to WAKE params."""
        return self.stage_audio.get(
            stage, self.stage_audio.get(SleepStage.WAKE, StageAudioParams())
        )

    def get_target_stage(self, progress: float) -> SleepStage:
        """Return the ideal stage for a given session *progress* in [0, 1].

        Progress is mapped linearly through the cumulative stage-target
        fractions in stage order (WAKE, N1, N2, N3, REM).
        """
        progress = max(0.0, min(1.0, progress))
        cumulative = 0.0
        for stage in (SleepStage.WAKE, SleepStage.N1, SleepStage.N2, SleepStage.N3, SleepStage.REM):
            cumulative += self.stage_targets.get(stage, 0.0)
            if progress <= cumulative:
                return stage
        return SleepStage.REM  # fallback at end of night

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the protocol to a plain dict."""
        return {
            "name": self.name,
            "description": self.description,
            "total_duration_min": self.total_duration_min,
            "stage_targets": {s.name: v for s, v in self.stage_targets.items()},
            "stage_audio": {
                s.name: {
                    "binaural_hz": a.binaural_hz,
                    "noise_color": a.noise_color,
                    "base_freq_hz": a.base_freq_hz,
                    "volume": a.volume,
                    "isochronic_hz": a.isochronic_hz,
                    "spatial_rotation": a.spatial_rotation,
                }
                for s, a in self.stage_audio.items()
            },
        }


# ---------------------------------------------------------------------------
# Protocol definitions
# ---------------------------------------------------------------------------


def _build_insomnia_relief() -> SleepProtocol:
    return SleepProtocol(
        name="insomnia_relief",
        description="Gradual descent from alpha to deep delta for chronic insomnia.",
        total_duration_min=480.0,
        stage_targets={
            SleepStage.WAKE: 0.05,
            SleepStage.N1: 0.10,
            SleepStage.N2: 0.45,
            SleepStage.N3: 0.25,
            SleepStage.REM: 0.15,
        },
        stage_audio={
            SleepStage.WAKE: StageAudioParams(
                binaural_hz=10.0,
                noise_color="pink",
                base_freq_hz=200.0,
                volume=0.4,
                isochronic_hz=0.0,
                spatial_rotation=5.0,
            ),
            SleepStage.N1: StageAudioParams(
                binaural_hz=6.0,
                noise_color="pink",
                base_freq_hz=180.0,
                volume=0.35,
                isochronic_hz=6.0,
                spatial_rotation=3.0,
            ),
            SleepStage.N2: StageAudioParams(
                binaural_hz=4.0,
                noise_color="brown",
                base_freq_hz=160.0,
                volume=0.30,
                isochronic_hz=4.0,
                spatial_rotation=2.0,
            ),
            SleepStage.N3: StageAudioParams(
                binaural_hz=2.0,
                noise_color="brown",
                base_freq_hz=140.0,
                volume=0.25,
                isochronic_hz=2.0,
                spatial_rotation=1.0,
            ),
            SleepStage.REM: StageAudioParams(
                binaural_hz=5.0,
                noise_color="pink",
                base_freq_hz=170.0,
                volume=0.30,
                isochronic_hz=0.0,
                spatial_rotation=4.0,
            ),
        },
    )


def _build_jet_lag_reset() -> SleepProtocol:
    return SleepProtocol(
        name="jet_lag_reset",
        description="Aggressive circadian resynchronisation with strong delta pulses.",
        total_duration_min=360.0,
        stage_targets={
            SleepStage.WAKE: 0.05,
            SleepStage.N1: 0.10,
            SleepStage.N2: 0.35,
            SleepStage.N3: 0.35,
            SleepStage.REM: 0.15,
        },
        stage_audio={
            SleepStage.WAKE: StageAudioParams(
                binaural_hz=8.0,
                noise_color="white",
                base_freq_hz=220.0,
                volume=0.45,
                isochronic_hz=8.0,
                spatial_rotation=6.0,
            ),
            SleepStage.N1: StageAudioParams(
                binaural_hz=5.0,
                noise_color="pink",
                base_freq_hz=190.0,
                volume=0.40,
                isochronic_hz=5.0,
                spatial_rotation=4.0,
            ),
            SleepStage.N2: StageAudioParams(
                binaural_hz=3.0,
                noise_color="brown",
                base_freq_hz=160.0,
                volume=0.35,
                isochronic_hz=3.0,
                spatial_rotation=2.0,
            ),
            SleepStage.N3: StageAudioParams(
                binaural_hz=1.5,
                noise_color="brown",
                base_freq_hz=130.0,
                volume=0.25,
                isochronic_hz=1.5,
                spatial_rotation=0.5,
            ),
            SleepStage.REM: StageAudioParams(
                binaural_hz=5.5,
                noise_color="pink",
                base_freq_hz=175.0,
                volume=0.30,
                isochronic_hz=0.0,
                spatial_rotation=3.0,
            ),
        },
    )


def _build_deep_sleep_boost() -> SleepProtocol:
    return SleepProtocol(
        name="deep_sleep_boost",
        description="Maximise N3 slow-wave sleep for physical recovery.",
        total_duration_min=480.0,
        stage_targets={
            SleepStage.WAKE: 0.03,
            SleepStage.N1: 0.07,
            SleepStage.N2: 0.30,
            SleepStage.N3: 0.40,
            SleepStage.REM: 0.20,
        },
        stage_audio={
            SleepStage.WAKE: StageAudioParams(
                binaural_hz=10.0,
                noise_color="pink",
                base_freq_hz=200.0,
                volume=0.40,
                isochronic_hz=0.0,
                spatial_rotation=4.0,
            ),
            SleepStage.N1: StageAudioParams(
                binaural_hz=6.0,
                noise_color="pink",
                base_freq_hz=180.0,
                volume=0.35,
                isochronic_hz=6.0,
                spatial_rotation=3.0,
            ),
            SleepStage.N2: StageAudioParams(
                binaural_hz=3.5,
                noise_color="brown",
                base_freq_hz=155.0,
                volume=0.30,
                isochronic_hz=3.5,
                spatial_rotation=1.5,
            ),
            SleepStage.N3: StageAudioParams(
                binaural_hz=1.0,
                noise_color="brown",
                base_freq_hz=120.0,
                volume=0.20,
                isochronic_hz=1.0,
                spatial_rotation=0.5,
            ),
            SleepStage.REM: StageAudioParams(
                binaural_hz=5.0,
                noise_color="pink",
                base_freq_hz=170.0,
                volume=0.30,
                isochronic_hz=0.0,
                spatial_rotation=3.5,
            ),
        },
    )


def _build_rem_enhancement() -> SleepProtocol:
    return SleepProtocol(
        name="rem_enhancement",
        description="Enhance REM sleep for memory consolidation and creativity.",
        total_duration_min=480.0,
        stage_targets={
            SleepStage.WAKE: 0.05,
            SleepStage.N1: 0.10,
            SleepStage.N2: 0.30,
            SleepStage.N3: 0.20,
            SleepStage.REM: 0.35,
        },
        stage_audio={
            SleepStage.WAKE: StageAudioParams(
                binaural_hz=10.0,
                noise_color="pink",
                base_freq_hz=200.0,
                volume=0.40,
                isochronic_hz=0.0,
                spatial_rotation=5.0,
            ),
            SleepStage.N1: StageAudioParams(
                binaural_hz=7.0,
                noise_color="pink",
                base_freq_hz=185.0,
                volume=0.35,
                isochronic_hz=7.0,
                spatial_rotation=3.5,
            ),
            SleepStage.N2: StageAudioParams(
                binaural_hz=4.0,
                noise_color="pink",
                base_freq_hz=165.0,
                volume=0.30,
                isochronic_hz=4.0,
                spatial_rotation=2.5,
            ),
            SleepStage.N3: StageAudioParams(
                binaural_hz=2.0,
                noise_color="brown",
                base_freq_hz=140.0,
                volume=0.25,
                isochronic_hz=2.0,
                spatial_rotation=1.0,
            ),
            SleepStage.REM: StageAudioParams(
                binaural_hz=6.0,
                noise_color="pink",
                base_freq_hz=180.0,
                volume=0.35,
                isochronic_hz=0.0,
                spatial_rotation=5.0,
            ),
        },
    )


def _build_shift_worker() -> SleepProtocol:
    return SleepProtocol(
        name="shift_worker",
        description="Compressed high-efficiency sleep for rotating shift schedules.",
        total_duration_min=360.0,
        stage_targets={
            SleepStage.WAKE: 0.05,
            SleepStage.N1: 0.05,
            SleepStage.N2: 0.35,
            SleepStage.N3: 0.35,
            SleepStage.REM: 0.20,
        },
        stage_audio={
            SleepStage.WAKE: StageAudioParams(
                binaural_hz=8.0,
                noise_color="brown",
                base_freq_hz=210.0,
                volume=0.45,
                isochronic_hz=8.0,
                spatial_rotation=6.0,
            ),
            SleepStage.N1: StageAudioParams(
                binaural_hz=5.0,
                noise_color="brown",
                base_freq_hz=180.0,
                volume=0.40,
                isochronic_hz=5.0,
                spatial_rotation=4.0,
            ),
            SleepStage.N2: StageAudioParams(
                binaural_hz=3.0,
                noise_color="brown",
                base_freq_hz=150.0,
                volume=0.30,
                isochronic_hz=3.0,
                spatial_rotation=2.0,
            ),
            SleepStage.N3: StageAudioParams(
                binaural_hz=1.0,
                noise_color="brown",
                base_freq_hz=120.0,
                volume=0.20,
                isochronic_hz=1.0,
                spatial_rotation=0.5,
            ),
            SleepStage.REM: StageAudioParams(
                binaural_hz=5.5,
                noise_color="pink",
                base_freq_hz=175.0,
                volume=0.30,
                isochronic_hz=0.0,
                spatial_rotation=3.5,
            ),
        },
    )


def _build_power_nap() -> SleepProtocol:
    return SleepProtocol(
        name="power_nap",
        description="20-minute alertness restoration; avoids deep sleep.",
        total_duration_min=20.0,
        stage_targets={
            SleepStage.WAKE: 0.15,
            SleepStage.N1: 0.35,
            SleepStage.N2: 0.40,
            SleepStage.N3: 0.05,
            SleepStage.REM: 0.05,
        },
        stage_audio={
            SleepStage.WAKE: StageAudioParams(
                binaural_hz=12.0,
                noise_color="white",
                base_freq_hz=220.0,
                volume=0.35,
                isochronic_hz=0.0,
                spatial_rotation=8.0,
            ),
            SleepStage.N1: StageAudioParams(
                binaural_hz=8.0,
                noise_color="pink",
                base_freq_hz=200.0,
                volume=0.30,
                isochronic_hz=8.0,
                spatial_rotation=5.0,
            ),
            SleepStage.N2: StageAudioParams(
                binaural_hz=6.0,
                noise_color="pink",
                base_freq_hz=180.0,
                volume=0.25,
                isochronic_hz=6.0,
                spatial_rotation=3.0,
            ),
            SleepStage.N3: StageAudioParams(
                binaural_hz=3.0,
                noise_color="brown",
                base_freq_hz=150.0,
                volume=0.20,
                isochronic_hz=3.0,
                spatial_rotation=1.0,
            ),
            SleepStage.REM: StageAudioParams(
                binaural_hz=7.0,
                noise_color="pink",
                base_freq_hz=190.0,
                volume=0.30,
                isochronic_hz=0.0,
                spatial_rotation=4.0,
            ),
        },
    )


# ---------------------------------------------------------------------------
# Protocol registry
# ---------------------------------------------------------------------------

PROTOCOL_REGISTRY: Dict[str, SleepProtocol] = {
    "insomnia_relief": _build_insomnia_relief(),
    "jet_lag_reset": _build_jet_lag_reset(),
    "deep_sleep_boost": _build_deep_sleep_boost(),
    "rem_enhancement": _build_rem_enhancement(),
    "shift_worker": _build_shift_worker(),
    "power_nap": _build_power_nap(),
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_protocol(name: str) -> SleepProtocol:
    """Look up a protocol by name.  Raises ``KeyError`` if not found."""
    return PROTOCOL_REGISTRY[name]


def list_protocols() -> List[str]:
    """Return a sorted list of all available protocol names."""
    return sorted(PROTOCOL_REGISTRY.keys())
