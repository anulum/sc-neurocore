# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPharmModel from former test_experiment.py

"""Focused suite: TestPharmModel from former test_experiment.py."""

from __future__ import annotations

from tests.test_bioware.experiment_support import *  # noqa: F403


class TestPharmModel:
    def test_no_drug(self) -> None:
        pm = PharmModel()
        assert pm.effective_gain(0.0) == 1.0

    def test_full_onset(self) -> None:
        pm = PharmModel(gain=2.0, onset_delay_s=10.0)
        pm.apply(0.0)
        assert pm.effective_gain(100.0) == 2.0

    def test_partial_onset(self) -> None:
        pm = PharmModel(gain=2.0, onset_delay_s=10.0)
        pm.apply(0.0)
        g = pm.effective_gain(5.0)  # half onset
        assert 1.0 < g < 2.0

    def test_modulate_spikes(self) -> None:
        pm = PharmModel(gain=0.0, onset_delay_s=0.0)  # TTX silencing
        pm.apply(0.0)
        counts = np.array([10, 20, 30])
        result = pm.modulate_spikes(counts, 100.0)
        assert np.all(result == 0)

    def test_modulate_spike_events_empty_input_returns_empty(self) -> None:
        pm = PharmModel(gain=2.0, onset_delay_s=0.0)
        pm.apply(0.0)

        assert pm.modulate_spike_events([], 1.0) == []

    def test_modulate_spike_events_zero_gain_returns_empty(self) -> None:
        pm = PharmModel(gain=0.0, onset_delay_s=0.0)
        pm.apply(0.0)
        spikes = [DetectedSpike(channel=0, timestamp_s=0.0, amplitude_uv=-40.0)]

        assert pm.modulate_spike_events(spikes, 1.0) == []

    def test_modulate_spike_events_inhibitory_preserves_response_span(self) -> None:
        pm = PharmModel(gain=0.5, onset_delay_s=0.0)
        pm.apply(0.0)
        spikes = [
            DetectedSpike(channel=i % 2, timestamp_s=i * 0.001, amplitude_uv=-40.0)
            for i in range(10)
        ]

        result = pm.modulate_spike_events(spikes, 1.0)

        assert len(result) == 5
        assert result[0].timestamp_s == pytest.approx(spikes[0].timestamp_s)
        assert result[-1].timestamp_s == pytest.approx(spikes[-1].timestamp_s)

    def test_modulate_spike_events_excitatory_inserts_within_observed_window(self) -> None:
        pm = PharmModel(gain=2.0, onset_delay_s=0.0)
        pm.apply(0.0)
        spikes = [
            DetectedSpike(channel=0, timestamp_s=0.000, amplitude_uv=-42.0),
            DetectedSpike(channel=1, timestamp_s=0.010, amplitude_uv=-38.0),
            DetectedSpike(channel=0, timestamp_s=0.020, amplitude_uv=-41.0),
        ]

        result = pm.modulate_spike_events(spikes, 1.0)
        timestamps = [s.timestamp_s for s in result]

        assert len(result) == 6
        assert timestamps == sorted(timestamps)
        assert min(timestamps) >= spikes[0].timestamp_s
        assert max(timestamps) <= spikes[-1].timestamp_s
        assert {s.channel for s in result} == {0, 1}

    def test_modulate_negative_gain_raises(self) -> None:
        with pytest.raises(ValueError, match="gain must be >= 0"):
            PharmModel(gain=-1.0, onset_delay_s=0.0)

    def test_modulate_unit_gain_preserves_events(self) -> None:
        pm = PharmModel(gain=1.0, onset_delay_s=0.0)
        pm.apply(0.0)
        spikes = [
            DetectedSpike(channel=0, timestamp_s=i * 0.001, amplitude_uv=-40.0) for i in range(4)
        ]
        result = pm.modulate_spike_events(spikes, 1.0)
        assert len(result) == 4  # gain 1.0 -> target count equals input count

    def test_modulate_excitatory_non_finite_timestamp_raises(self) -> None:
        with pytest.raises(ValueError, match="timestamp_s must be finite"):
            DetectedSpike(channel=0, timestamp_s=float("inf"), amplitude_uv=-40.0)

    def test_modulate_excitatory_single_spike_clones(self) -> None:
        pm = PharmModel(gain=3.0, onset_delay_s=0.0)
        pm.apply(0.0)
        spikes = [DetectedSpike(channel=0, timestamp_s=0.005, amplitude_uv=-40.0)]
        result = pm.modulate_spike_events(spikes, 1.0)
        assert len(result) == 3  # single observed spike plus two clones

    def test_quantile_indices_edge_counts(self) -> None:
        from sc_neurocore.bioware.bioware import _quantile_indices

        assert _quantile_indices(5, 0) == []  # non-positive target keeps no events
        assert _quantile_indices(3, 5) == [0, 1, 2]  # target >= n keeps all
        assert _quantile_indices(5, 1) == [0]  # a single sample takes the head
