# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stable biological-interface data contracts

"""Stable data contracts for the biological-hardware interface."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from .bioware_validation import (
    require_finite,
    require_nonnegative,
    require_nonnegative_int,
    require_positive,
    require_positive_int,
    validate_binary_bitstream,
)


class MEALayout(Enum):
    """Standard MEA electrode layouts."""

    MEA_60 = "60ch"
    MEA_120 = "120ch"
    MEA_256 = "256ch"
    MEA_4096 = "4096ch"
    CUSTOM = "custom"


@dataclass
class MEAConfig:
    """Multi-electrode array configuration."""

    layout: MEALayout = MEALayout.MEA_60
    num_channels: int = 60
    sample_rate_hz: float = 20_000.0
    voltage_gain: float = 1000.0
    noise_floor_uv: float = 5.0
    spike_threshold_sigma: float = 5.0
    electrode_pitch_um: float = 200.0

    def __post_init__(self) -> None:
        """Validate physical and acquisition configuration boundaries."""
        if not isinstance(self.layout, MEALayout):
            raise TypeError("layout must be a MEALayout")
        require_positive_int(self.num_channels, "num_channels")
        require_positive(self.sample_rate_hz, "sample_rate_hz")
        require_positive(self.voltage_gain, "voltage_gain")
        require_nonnegative(self.noise_floor_uv, "noise_floor_uv")
        require_positive(self.spike_threshold_sigma, "spike_threshold_sigma")
        require_positive(self.electrode_pitch_um, "electrode_pitch_um")

    @classmethod
    def from_layout(cls, layout: MEALayout) -> MEAConfig:
        """Create a configuration preset for a standard MEA layout.

        Parameters
        ----------
        layout:
            Standard electrode layout whose channel count and pitch should seed
            the returned configuration.

        Returns
        -------
        MEAConfig
            Configuration with the requested layout and canonical channel/pitch
            preset while retaining the default sampling and detector gains.
        """
        if not isinstance(layout, MEALayout):
            raise TypeError("layout must be a MEALayout")
        presets: Dict[MEALayout, Dict[str, Any]] = {
            MEALayout.MEA_60: dict(num_channels=60, electrode_pitch_um=200.0),
            MEALayout.MEA_120: dict(num_channels=120, electrode_pitch_um=100.0),
            MEALayout.MEA_256: dict(num_channels=256, electrode_pitch_um=100.0),
            MEALayout.MEA_4096: dict(num_channels=4096, electrode_pitch_um=17.5),
            MEALayout.CUSTOM: dict(num_channels=60, electrode_pitch_um=200.0),
        }
        return cls(layout=layout, **presets[layout])


# ── Spike Detection + Sorting ────────────────────────────────────────


@dataclass
class DetectedSpike:
    """One detected spike event from MEA data."""

    channel: int
    timestamp_s: float
    amplitude_uv: float
    unit_id: int = 0  # cluster assignment
    waveform: Optional[np.ndarray[Any, Any]] = None

    def __post_init__(self) -> None:
        """Validate spike identity, timing, amplitude, and optional waveform."""
        require_nonnegative_int(self.channel, "channel")
        require_nonnegative(self.timestamp_s, "timestamp_s")
        require_finite(self.amplitude_uv, "amplitude_uv")
        require_nonnegative_int(self.unit_id, "unit_id")
        if self.waveform is None:
            return
        if not isinstance(self.waveform, np.ndarray):
            raise TypeError("waveform must be a NumPy array when provided")
        if self.waveform.ndim != 1 or self.waveform.size == 0:
            raise ValueError("waveform must be a non-empty one-dimensional array")
        if not np.issubdtype(self.waveform.dtype, np.number):
            raise TypeError("waveform must have a numeric dtype")
        if not np.all(np.isfinite(self.waveform)):
            raise ValueError("waveform must contain only finite values")


@dataclass
class AEREvent:
    """Address-Event Representation packet.

    Compatible with sc_aer_encoder.v format:
    {valid, neuron_id, timestamp}
    """

    neuron_id: int
    timestamp: int  # clock ticks (not real time)
    valid: bool = True
    weight: int = 256  # Q8.8 = 1.0

    def __post_init__(self) -> None:
        """Validate the maintained unsigned 16-bit AER packet fields."""
        require_nonnegative_int(self.neuron_id, "neuron_id")
        require_nonnegative_int(self.timestamp, "timestamp")
        if self.timestamp > 0xFFFF:
            raise ValueError("timestamp must fit the 16-bit AER field")
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a bool")
        require_nonnegative_int(self.weight, "weight")
        if self.weight > 0xFFFF:
            raise ValueError("weight must fit an unsigned 16-bit Q8.8 field")


class StimProtocol(Enum):
    """Optogenetic stimulation protocols."""

    CONSTANT = "constant"
    PULSED = "pulsed"
    GRADED = "graded"
    PATTERN = "pattern"


@dataclass
class OptogeneticPulse:
    """One optical stimulation pulse."""

    channel: int
    onset_ms: float
    duration_ms: float
    intensity_mw_mm2: float = 1.0
    wavelength_nm: int = 470  # blue (ChR2)
    illuminated_area_mm2: float = 1.0

    def __post_init__(self) -> None:
        """Validate timing, irradiance, wavelength, and illuminated area."""
        require_nonnegative_int(self.channel, "channel")
        require_nonnegative(self.onset_ms, "onset_ms")
        require_positive(self.duration_ms, "duration_ms")
        require_nonnegative(self.intensity_mw_mm2, "intensity_mw_mm2")
        require_positive_int(self.wavelength_nm, "wavelength_nm")
        require_positive(self.illuminated_area_mm2, "illuminated_area_mm2")

    @property
    def power_mw(self) -> float:
        """Return optical power as irradiance multiplied by illuminated area."""
        return self.intensity_mw_mm2 * self.illuminated_area_mm2


@dataclass
class BioHybridFrameResult:
    """Strictly typed output packet detailing a full closed-loop step.

    Behaves both as a dataclass (``result.round``) and, for backward
    compatibility with pre-dataclass callers, as a mapping view of its
    fields (``result["round"]``, ``"latency_us" in result``,
    ``dict(result)``). The mapping surface is read-only.
    """

    round: int
    num_spikes: int
    num_aer_events: int
    num_bitstreams: int
    num_opto_pulses: int
    latency_us: float
    health: Dict[str, Any]
    spikes: List[DetectedSpike]
    aer_events: List[AEREvent]
    bitstreams: Dict[int, np.ndarray[Any, Any]]
    opto_pulses: List[OptogeneticPulse]

    def __post_init__(self) -> None:
        """Validate counts and payload cardinalities for one closed-loop frame."""
        require_nonnegative_int(self.round, "round")
        for name, value in (
            ("num_spikes", self.num_spikes),
            ("num_aer_events", self.num_aer_events),
            ("num_bitstreams", self.num_bitstreams),
            ("num_opto_pulses", self.num_opto_pulses),
        ):
            require_nonnegative_int(value, name)
        require_nonnegative(self.latency_us, "latency_us")
        if self.num_spikes != len(self.spikes):
            raise ValueError("num_spikes must equal len(spikes)")
        if self.num_aer_events != len(self.aer_events):
            raise ValueError("num_aer_events must equal len(aer_events)")
        if self.num_bitstreams != len(self.bitstreams):
            raise ValueError("num_bitstreams must equal len(bitstreams)")
        if self.num_opto_pulses != len(self.opto_pulses):
            raise ValueError("num_opto_pulses must equal len(opto_pulses)")
        for neuron_id, bitstream in self.bitstreams.items():
            require_nonnegative_int(neuron_id, "bitstream neuron_id")
            validate_binary_bitstream(
                bitstream,
                name=f"bitstreams[{neuron_id}]",
                allow_empty=True,
            )

    def __getitem__(self, key: str) -> Any:
        """Return a dataclass field through the legacy mapping interface.

        Parameters
        ----------
        key:
            Public dataclass field name to read.

        Returns
        -------
        Any
            The underlying field value, preserving object identity for mutable
            payloads such as ``health`` and ``bitstreams``.

        Raises
        ------
        KeyError
            If ``key`` is not a public field name.
        """
        if not isinstance(key, str) or key.startswith("_"):
            raise KeyError(key)
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __contains__(self, key: object) -> bool:
        """Return whether ``key`` names a public result field.

        Parameters
        ----------
        key:
            Candidate mapping key.

        Returns
        -------
        bool
            ``True`` only for string keys matching declared dataclass fields.
        """
        if not isinstance(key, str):
            return False
        return key in {f.name for f in fields(self)}

    def keys(self) -> List[str]:
        """Return the mapping-view field names in dataclass declaration order.

        Returns
        -------
        list[str]
            Public field names accepted by ``__getitem__`` and ``__contains__``.
        """
        return [f.name for f in fields(self)]
