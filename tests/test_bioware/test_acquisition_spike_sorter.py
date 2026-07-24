# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeSorter from former test_acquisition.py

"""Focused suite: TestSpikeSorter from former test_acquisition.py."""

from __future__ import annotations

from tests.test_bioware.acquisition_support import *  # noqa: F403


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
