# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pharmacology and multi-well experiment support

"""Pharmacology and multi-well experiment support."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, List, Optional

import numpy as np

from .bioware_contracts import DetectedSpike, MEAConfig, MEALayout
from .bioware_validation import (
    require_finite,
    require_nonnegative,
    require_nonnegative_int,
    require_positive,
)


@dataclass
class PharmModel:
    """Simulates effect of pharmacological agents on spike rate.

    Models excitatory (e.g., bicuculline) or inhibitory (e.g., TTX) agents
    as gain factors on firing rate.
    """

    agent_name: str = "none"
    gain: float = 1.0  # >1 = excitatory, <1 = inhibitory, 0 = silencing
    onset_delay_s: float = 30.0
    wash_time_s: float = 120.0
    _applied_at: float = -1.0

    def __post_init__(self) -> None:
        """Validate the pharmacological gain and experiment-time constants."""
        if not self.agent_name or not self.agent_name.strip():
            raise ValueError("agent_name must not be empty")
        require_nonnegative(self.gain, "gain")
        require_nonnegative(self.onset_delay_s, "onset_delay_s")
        require_positive(self.wash_time_s, "wash_time_s")
        require_finite(self._applied_at, "_applied_at")
        if self._applied_at < -1.0:
            raise ValueError("_applied_at must be -1 or a non-negative timestamp")

    def apply(self, t_current_s: float) -> None:
        """Mark the pharmacological agent as applied at the current time.

        Parameters
        ----------
        t_current_s:
            Experiment time in seconds used as the onset reference for
            subsequent gain interpolation.
        """
        require_nonnegative(t_current_s, "t_current_s")
        self._applied_at = t_current_s

    def effective_gain(self, t_current_s: float) -> float:
        """Return the active firing-rate gain at an experiment timestamp.

        Parameters
        ----------
        t_current_s:
            Experiment time in seconds.

        Returns
        -------
        float
            ``1.0`` before application, a linearly interpolated onset gain
            during ``onset_delay_s``, or the configured steady-state gain after
            onset.
        """
        require_nonnegative(t_current_s, "t_current_s")
        if self._applied_at < 0:
            return 1.0
        elapsed = t_current_s - self._applied_at
        if elapsed < 0.0:
            raise ValueError("t_current_s must not precede the application time")
        if elapsed < self.onset_delay_s:
            frac = elapsed / self.onset_delay_s
            return 1.0 + frac * (self.gain - 1.0)
        return self.gain

    def modulate_spikes(
        self, spike_counts: np.ndarray[Any, Any], t_current_s: float
    ) -> np.ndarray[Any, Any]:
        """Modulate spike counts by pharmacological gain."""
        if not isinstance(spike_counts, np.ndarray):
            raise TypeError("spike_counts must be a NumPy array")
        if spike_counts.ndim != 1:
            raise ValueError("spike_counts must be one-dimensional")
        if not np.issubdtype(spike_counts.dtype, np.number):
            raise TypeError("spike_counts must have a numeric dtype")
        if not np.all(np.isfinite(spike_counts)) or np.any(spike_counts < 0):
            raise ValueError("spike_counts must contain finite non-negative values")
        g = self.effective_gain(t_current_s)
        return np.round(spike_counts * g).astype(int)

    def modulate_spike_events(
        self,
        spikes: List[DetectedSpike],
        t_current_s: float,
    ) -> List[DetectedSpike]:
        """Apply pharmacological rate gain to spike events.

        Inhibitory gains deterministically thin events across the observed
        response span instead of truncating the head of the frame. Excitatory
        gains preserve observed events and insert synthetic events inside the
        observed temporal support, using nearest observed spikes as channel,
        unit, amplitude, and waveform templates.
        """
        require_nonnegative(t_current_s, "t_current_s")
        if not spikes:
            return []

        gain = self.effective_gain(t_current_s)
        if not math.isfinite(gain) or gain < 0.0:
            raise ValueError("pharmacological gain must be finite and >= 0")

        ordered = sorted(spikes, key=lambda s: (s.timestamp_s, s.channel, s.unit_id))
        target_count = int(round(len(ordered) * gain))
        if target_count <= 0:
            return []
        if target_count == len(ordered):
            return list(ordered)
        if target_count < len(ordered):
            indices = _quantile_indices(len(ordered), target_count)
            return [ordered[i] for i in indices]

        extra = target_count - len(ordered)
        timestamps = np.array([s.timestamp_s for s in ordered], dtype=float)
        if not np.all(np.isfinite(timestamps)):
            raise ValueError("detected spike timestamps must be finite")

        if len(ordered) == 1 or timestamps[-1] <= timestamps[0]:
            synthetic = [
                _clone_spike(ordered[0], timestamp_s=float(timestamps[0])) for _ in range(extra)
            ]
        else:
            insert_times = np.linspace(timestamps[0], timestamps[-1], extra + 2)[1:-1]
            synthetic = []
            for t in insert_times:
                # insert_times are strictly interior to (timestamps[0],
                # timestamps[-1]) — the linspace endpoints are dropped — so for
                # t < timestamps[-1] a left-side searchsorted always yields
                # idx <= len(ordered) - 1; no upper clamp is reachable.
                idx = int(np.searchsorted(timestamps, t, side="left"))
                if idx > 0 and abs(timestamps[idx - 1] - t) <= abs(timestamps[idx] - t):
                    idx -= 1
                synthetic.append(_clone_spike(ordered[idx], timestamp_s=float(t)))

        return sorted([*ordered, *synthetic], key=lambda s: (s.timestamp_s, s.channel, s.unit_id))


def _quantile_indices(n_items: int, target_count: int) -> List[int]:
    require_nonnegative_int(n_items, "n_items")
    require_nonnegative_int(target_count, "target_count")
    if n_items == 0 and target_count > 0:
        raise ValueError("target_count must be zero when n_items is zero")
    if target_count <= 0:
        return []
    if target_count >= n_items:
        return list(range(n_items))
    if target_count == 1:
        return [0]
    return [int(i) for i in np.rint(np.linspace(0, n_items - 1, target_count)).astype(int)]


def _clone_spike(template: DetectedSpike, *, timestamp_s: float) -> DetectedSpike:
    require_nonnegative(timestamp_s, "timestamp_s")
    waveform = None if template.waveform is None else template.waveform.copy()
    return replace(template, timestamp_s=timestamp_s, waveform=waveform)


@dataclass
class WellConfig:
    """One well in a multi-well MEA plate."""

    well_id: str
    mea_config: MEAConfig
    culture_type: str = "cortical"
    passage_number: int = 0

    def __post_init__(self) -> None:
        """Validate well identity, culture label, passage, and MEA config."""
        if not self.well_id or not self.well_id.strip():
            raise ValueError("well_id must not be empty")
        if not isinstance(self.mea_config, MEAConfig):
            raise TypeError("mea_config must be an MEAConfig")
        if not self.culture_type or not self.culture_type.strip():
            raise ValueError("culture_type must not be empty")
        require_nonnegative_int(self.passage_number, "passage_number")

    @property
    def label(self) -> str:
        """Return the stable plate label for this well.

        Returns
        -------
        str
            Identifier combining well ID, culture type, and passage number.
        """
        return f"{self.well_id}_{self.culture_type}_P{self.passage_number}"


@dataclass
class MultiWellPlate:
    """Multi-well plate (e.g., 6/24/48/96-well MEA plate)."""

    wells: List[WellConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate well types and unique well identifiers."""
        seen: set[str] = set()
        for well in self.wells:
            if not isinstance(well, WellConfig):
                raise TypeError("wells must contain WellConfig instances")
            if well.well_id in seen:
                raise ValueError(f"duplicate well_id: {well.well_id}")
            seen.add(well.well_id)

    def add_well(self, well: WellConfig) -> None:
        """Append a well configuration to the plate.

        Parameters
        ----------
        well:
            Well metadata and MEA configuration to append.
        """
        if not isinstance(well, WellConfig):
            raise TypeError("well must be a WellConfig")
        if self.get_well(well.well_id) is not None:
            raise ValueError(f"duplicate well_id: {well.well_id}")
        self.wells.append(well)

    @classmethod
    def standard_6_well(cls, layout: MEALayout = MEALayout.MEA_60) -> MultiWellPlate:
        """Construct a six-well plate with uniform MEA layout presets.

        Parameters
        ----------
        layout:
            MEA layout preset used for each generated well.

        Returns
        -------
        MultiWellPlate
            Plate containing wells ``W1`` through ``W6``.
        """
        if not isinstance(layout, MEALayout):
            raise TypeError("layout must be a MEALayout")
        plate = cls()
        for i in range(6):
            plate.add_well(
                WellConfig(
                    well_id=f"W{i + 1}",
                    mea_config=MEAConfig.from_layout(layout),
                )
            )
        return plate

    @property
    def num_wells(self) -> int:
        """Return the number of configured wells.

        Returns
        -------
        int
            Length of the plate's well list.
        """
        return len(self.wells)

    def get_well(self, well_id: str) -> Optional[WellConfig]:
        """Return a well by identifier.

        Parameters
        ----------
        well_id:
            Identifier such as ``"W1"``.

        Returns
        -------
        WellConfig | None
            Matching well configuration, or ``None`` when the plate does not
            contain ``well_id``.
        """
        if not well_id or not well_id.strip():
            raise ValueError("well_id must not be empty")
        return next((w for w in self.wells if w.well_id == well_id), None)
