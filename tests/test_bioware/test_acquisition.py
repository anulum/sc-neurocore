# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bioware acquisition tests

"""Tests for MEA acquisition, spike detection, sorting, and artifacts."""

from __future__ import annotations

import sys
from collections.abc import MutableMapping
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.bioware.bioware import (
    ArtifactRejector,
    DetectedSpike,
    MEAConfig,
    SpikeDetector,
    SpikeSorter,
)


FloatArray = npt.NDArray[np.float64]


def _synth_voltage(
    n_samples: int = 1000,
    n_channels: int = 10,
    seed: int = 42,
) -> FloatArray:
    """Generate synthetic MEA voltage data with embedded spikes."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 5, size=(n_samples, n_channels))
    for i in range(0, n_samples, 200):
        data[i, 0] = -80.0
        if i + 50 < n_samples:
            data[i + 50, 3] = -60.0
    return data


class TestSpikeDetector:
    def test_estimate_noise(self) -> None:
        cfg = MEAConfig(num_channels=10)
        det = SpikeDetector(config=cfg)
        data = _synth_voltage()
        noise = det.estimate_noise(data)
        assert len(noise) == 10
        assert np.all(noise > 0)

    def test_detect_spikes(self) -> None:
        cfg = MEAConfig(num_channels=10, spike_threshold_sigma=3.0)
        det = SpikeDetector(config=cfg)
        data = _synth_voltage()
        spikes = det.detect(data)
        assert len(spikes) > 0

    def test_spike_channels(self) -> None:
        cfg = MEAConfig(num_channels=10, spike_threshold_sigma=3.0)
        det = SpikeDetector(config=cfg)
        data = _synth_voltage()
        spikes = det.detect(data)
        channels = set(s.channel for s in spikes)
        assert 0 in channels  # We injected spikes on channel 0

    def test_spike_has_timestamp(self) -> None:
        cfg = MEAConfig(num_channels=10)
        det = SpikeDetector(config=cfg)
        spikes = det.detect(_synth_voltage())
        for s in spikes:
            assert s.timestamp_s >= 0

    def test_edge_spike_waveform_is_padded_to_fixed_length(self) -> None:
        # A spike close to the start of the recording yields a truncated raw
        # snippet that must be left-padded to the fixed snippet length.
        cfg = MEAConfig(num_channels=1, sample_rate_hz=20000.0, spike_threshold_sigma=3.0)
        det = SpikeDetector(config=cfg, refractory_samples=0)
        data = np.random.default_rng(0).normal(0.0, 1.0, size=(2000, 1))
        data[5, 0] = -100.0  # strong spike within half a snippet of the edge
        spikes = det.detect(data)
        assert spikes, "edge spike should be detected"
        target_len = 2 * int(2.0 * 20000.0 / 2000.0)
        for spike in spikes:
            assert spike.waveform is not None
            assert len(spike.waveform) == target_len


# ── MEAToAERTranscoder Tests ─────────────────────────────────────────


class TestRefractoryPeriod:
    def test_refractory_reduces_spikes(self) -> None:
        cfg = MEAConfig(num_channels=10, spike_threshold_sigma=3.0)
        det_no_ref = SpikeDetector(config=cfg, refractory_samples=0)
        det_with_ref = SpikeDetector(config=cfg, refractory_samples=50)
        data = _synth_voltage()
        spikes_no = det_no_ref.detect(data)
        spikes_yes = det_with_ref.detect(data)
        assert len(spikes_yes) <= len(spikes_no)

    def test_refractory_zero_matches_original(self) -> None:
        cfg = MEAConfig(num_channels=10, spike_threshold_sigma=3.0)
        # refractory_samples=0 means no refractory filter
        det = SpikeDetector(config=cfg, refractory_samples=0)
        data = _synth_voltage()
        spikes = det.detect(data)
        assert len(spikes) > 0


# ── Optogenetic Safety Tests ─────────────────────────────────────────


class TestSpikeSorter:
    def test_fit_without_sklearn_raises_actionable_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module_cache = cast(MutableMapping[str, object], sys.modules)
        monkeypatch.setitem(module_cache, "sklearn.cluster", None)
        monkeypatch.setitem(module_cache, "sklearn.decomposition", None)
        spikes = [
            DetectedSpike(
                channel=0,
                timestamp_s=float(i) * 0.001,
                amplitude_uv=-40.0,
                waveform=np.array([-1.0, -2.0, -1.0], dtype=np.float64),
            )
            for i in range(2)
        ]

        with pytest.raises(ImportError, match="requires scikit-learn"):
            SpikeSorter(num_units=2).fit(spikes)

    def test_fit_and_assign(self) -> None:
        # SpikeSorter clusters on the waveform shape, not the scalar
        # amplitude — construct three clearly distinct waveforms so PCA +
        # KMeans have enough variance to separate them into distinct
        # units. Each shape is a different half-sine + noise-free profile.
        pytest.importorskip("sklearn")  # cluster backend
        rng = np.random.default_rng(seed=42)
        t = np.linspace(0.0, 1.0, 32, dtype=np.float64)
        shapes = {
            0: -30.0 * np.sin(np.pi * t),  # shallow early peak
            1: -60.0 * np.sin(np.pi * t) ** 2,  # deeper, broader
            2: -90.0 * np.exp(-(((t - 0.5) / 0.1) ** 2)),  # narrow spike
        }
        spikes: list[DetectedSpike] = []
        for unit, base in shapes.items():
            for rep in range(6):  # 6 copies per unit = 18 total
                wf = base + rng.normal(0.0, 1.5, size=base.shape)
                spikes.append(
                    DetectedSpike(
                        channel=0,
                        timestamp_s=rep * 0.01 + unit * 0.1,
                        amplitude_uv=float(wf.min()),
                        waveform=wf,
                    )
                )
        sorter = SpikeSorter(num_units=3)
        sorter.fit(spikes)
        sorted_spikes = sorter.assign(spikes)
        assert len(sorted_spikes) == len(spikes)
        units = {s.unit_id for s in sorted_spikes}
        assert len(units) > 1, f"PCA+KMeans produced a single cluster: {units}"

    def test_no_fit_returns_original(self) -> None:
        sorter = SpikeSorter(num_units=3)
        spikes = [DetectedSpike(channel=0, timestamp_s=0.0, amplitude_uv=-50)]
        result = sorter.assign(spikes)
        assert result[0].unit_id == 0

    def test_empty_spikes(self) -> None:
        sorter = SpikeSorter()
        sorter.fit([])
        result = sorter.assign([])
        assert len(result) == 0

    def test_assign_passes_through_spike_without_waveform(self) -> None:
        pytest.importorskip("sklearn")
        rng = np.random.default_rng(seed=7)
        t = np.linspace(0.0, 1.0, 32, dtype=np.float64)
        shapes = [-30.0 * np.sin(np.pi * t), -60.0 * np.sin(np.pi * t) ** 2]
        train = [
            DetectedSpike(
                channel=0,
                timestamp_s=0.01 * rep + 0.1 * unit,
                amplitude_uv=float(base.min()),
                waveform=base + rng.normal(0.0, 1.5, size=base.shape),
            )
            for unit, base in enumerate(shapes)
            for rep in range(6)
        ]
        sorter = SpikeSorter(num_units=2)
        sorter.fit(train)
        # A spike with no recorded waveform cannot be projected and is passed
        # through unchanged.
        no_wave = DetectedSpike(channel=1, timestamp_s=0.5, amplitude_uv=-50.0, waveform=None)
        result = sorter.assign([no_wave])
        assert result == [no_wave]


# ── LFP Extraction Tests (Gap 2) ───────────────────────────────────────


class TestArtifactRejection:
    def test_blanking(self) -> None:
        data = np.ones((1000, 5))
        ar = ArtifactRejector(blanking_pre_ms=0.5, blanking_post_ms=2.0)
        blanked = ar.blank(data, stim_times_s=[0.025], sample_rate_hz=20000.0)
        # Centre at sample 500, pre=10 post=40 → blanked
        assert blanked[500, 0] == 0.0

    def test_no_stim_no_blanking(self) -> None:
        data = np.ones((100, 3))
        ar = ArtifactRejector()
        blanked = ar.blank(data, stim_times_s=[], sample_rate_hz=20000.0)
        np.testing.assert_array_equal(blanked, data)


# ── Bio Audit Log Tests (Gap 8) ────────────────────────────────────────
