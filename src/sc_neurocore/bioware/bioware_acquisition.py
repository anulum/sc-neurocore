# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MEA acquisition, spike detection, and artifact handling

"""MEA acquisition, spike detection, sorting, and artifact handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, cast

import numpy as np

from .bioware_contracts import DetectedSpike, MEAConfig
from .bioware_validation import (
    require_nonnegative,
    require_nonnegative_int,
    require_positive,
    require_positive_int,
    validate_voltage_matrix,
)


@dataclass
class SpikeDetector:
    """Threshold-based spike detector for MEA voltage traces.

    Uses adaptive threshold: threshold = mean ± sigma * noise_estimate
    where noise_estimate = median(|x|) / 0.6745 (robust RMS).
    Supports configurable refractory period to prevent double-counting.
    """

    config: MEAConfig
    refractory_samples: int = 30
    _noise_estimates: Optional[np.ndarray[Any, Any]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate detector configuration and refractory interval."""
        if not isinstance(self.config, MEAConfig):
            raise TypeError("config must be an MEAConfig")
        require_nonnegative_int(self.refractory_samples, "refractory_samples")

    def estimate_noise(self, voltage_data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Estimate per-channel noise from voltage data.

        Uses median absolute deviation (MAD) for robustness against spikes.
        voltage_data: shape (num_samples, num_channels)
        """
        validate_voltage_matrix(voltage_data, expected_channels=self.config.num_channels)
        mad: np.ndarray[Any, Any] = np.median(np.abs(voltage_data), axis=0) / 0.6745
        self._noise_estimates = mad
        return mad

    def detect(
        self, voltage_data: np.ndarray[Any, Any], snippet_ms: float = 2.0
    ) -> List[DetectedSpike]:
        """Detect spikes in multi-channel voltage data.

        voltage_data: shape (num_samples, num_channels)
        Returns list of DetectedSpike events.
        """
        validate_voltage_matrix(voltage_data, expected_channels=self.config.num_channels)
        require_positive(snippet_ms, "snippet_ms")
        n_samples, n_channels = voltage_data.shape
        if self._noise_estimates is None:
            self.estimate_noise(voltage_data)
        noise_estimates = cast(np.ndarray[Any, Any], self._noise_estimates)

        spikes = []
        dt = 1.0 / self.config.sample_rate_hz
        sigma = self.config.spike_threshold_sigma
        half = int(snippet_ms * self.config.sample_rate_hz / 2000.0)
        if half < 1:
            raise ValueError("snippet_ms is shorter than one sample on each side")

        for ch in range(n_channels):
            threshold = sigma * noise_estimates[ch]
            above = np.abs(voltage_data[:, ch]) > threshold
            crossings = np.where(np.diff(above.astype(int)) == 1)[0]
            last_spike_idx = -self.refractory_samples - 1
            for idx in crossings:
                if idx - last_spike_idx < self.refractory_samples:
                    continue
                last_spike_idx = idx
                amp = float(voltage_data[idx, ch])
                ts = idx * dt

                # Extract waveform snippet
                start = max(0, idx - half)
                end = min(n_samples, idx + half)

                # Pad if too close to edges. The raw slice length is at most
                # 2*half == target_len (it is min(n, idx+half) - max(0, idx-half)),
                # and for an edge spike (idx < half) the slice starts at 0 so
                # pad_before = half - idx exactly closes the gap: pad_after is
                # never negative and the padded waveform is exactly target_len.
                raw_wave = voltage_data[start:end, ch].copy()
                target_len = int(2 * half)
                if len(raw_wave) < target_len:
                    pad_before = max(0, half - idx)
                    pad_after = max(0, target_len - len(raw_wave) - pad_before)
                    raw_wave = np.pad(raw_wave, (pad_before, pad_after), "constant")

                spikes.append(
                    DetectedSpike(
                        channel=ch,
                        timestamp_s=ts,
                        amplitude_uv=amp,
                        unit_id=ch,
                        waveform=raw_wave,
                    )
                )
        return spikes


@dataclass
class SpikeSorter:
    """Research spike sorter using PCA feature extraction and K-Means clustering.

    Projects uniform waveforms onto their dominant principal components before
    clustering them into units. Fitting requires the optional ``scikit-learn``
    dependency; incomplete waveform sets remain explicitly unassigned.
    """

    num_units: int = 4
    n_components: int = 3
    random_state: int = 0
    _pca: Any = field(default=None, repr=False)
    _kmeans: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate cluster count, projection width, and deterministic seed."""
        require_positive_int(self.num_units, "num_units")
        require_positive_int(self.n_components, "n_components")
        require_nonnegative_int(self.random_state, "random_state")

    def fit(self, spikes: List[DetectedSpike]) -> None:
        """Fit PCA and KMeans models sequentially on available waveforms.

        Silently no-ops (leaves ``_pca``/``_kmeans`` as ``None``) when
        fewer than ``num_units`` waveforms are present — sklearn is only
        imported in the path that actually needs it, so empty or
        amplitude-only spike lists don't require scikit-learn.
        """
        waveforms = [s.waveform for s in spikes if s.waveform is not None]
        if len(waveforms) < self.num_units:
            self._pca = None
            self._kmeans = None
            return

        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
        except ImportError as exc:
            raise ImportError(
                "SpikeSorter.fit requires scikit-learn to cluster waveforms. "
                "Install with `pip install scikit-learn` or "
                "`pip install 'sc-neurocore[bioware]'`."
            ) from exc

        waveform_lengths = {waveform.shape for waveform in waveforms}
        if len(waveform_lengths) != 1:
            raise ValueError("all spike waveforms must have the same shape")
        waves_array = np.vstack(waveforms)
        self._pca = PCA(n_components=min(self.n_components, len(waveforms), waves_array.shape[1]))
        features = self._pca.fit_transform(waves_array)

        self._kmeans = KMeans(
            n_clusters=self.num_units,
            n_init=10,
            random_state=self.random_state,
        )
        self._kmeans.fit(features)

    def assign(self, spikes: List[DetectedSpike]) -> List[DetectedSpike]:
        """Assign cluster IDs based on PCA feature projections."""
        if self._pca is None or self._kmeans is None:
            return spikes

        result = []
        for s in spikes:
            if s.waveform is None:
                result.append(s)
                continue

            expected_features = int(self._pca.n_features_in_)
            if s.waveform.size != expected_features:
                raise ValueError(
                    f"spike waveform has {s.waveform.size} samples; expected {expected_features}"
                )

            features = self._pca.transform(s.waveform.reshape(1, -1))
            unit = int(self._kmeans.predict(features)[0])

            result.append(
                DetectedSpike(
                    channel=s.channel,
                    timestamp_s=s.timestamp_s,
                    amplitude_uv=s.amplitude_uv,
                    unit_id=unit,
                    waveform=s.waveform,
                )
            )
        return result


@dataclass
class ArtifactRejector:
    """Blanks stimulation artifacts from voltage data.

    Zeros the voltage trace in a window around each stimulation onset.
    """

    blanking_pre_ms: float = 0.5
    blanking_post_ms: float = 2.0

    def __post_init__(self) -> None:
        """Validate non-negative artifact-blanking intervals."""
        require_nonnegative(self.blanking_pre_ms, "blanking_pre_ms")
        require_nonnegative(self.blanking_post_ms, "blanking_post_ms")

    def blank(
        self,
        voltage_data: np.ndarray[Any, Any],
        stim_times_s: List[float],
        sample_rate_hz: float,
    ) -> np.ndarray[Any, Any]:
        """Return voltage data with stimulus artifacts blanked."""
        validate_voltage_matrix(voltage_data)
        require_positive(sample_rate_hz, "sample_rate_hz")
        result = voltage_data.copy()
        pre_samples = int(self.blanking_pre_ms * sample_rate_hz / 1000.0)
        post_samples = int(self.blanking_post_ms * sample_rate_hz / 1000.0)

        duration_s = result.shape[0] / sample_rate_hz
        for t_s in stim_times_s:
            require_nonnegative(t_s, "stimulus time")
            if t_s >= duration_s:
                raise ValueError("stimulus time must fall inside the voltage frame")
            center = int(t_s * sample_rate_hz)
            start = max(0, center - pre_samples)
            end = min(result.shape[0], center + post_samples)
            result[start:end, :] = 0.0
        return result
