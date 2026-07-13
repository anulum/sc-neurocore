# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bioware experiment and audit tests

"""Tests for pharmacology, multi-well experiments, and audit records."""

from __future__ import annotations

import hashlib
import json
import numpy as np
import pytest

from sc_neurocore.bioware.bioware import (
    BioAuditEntry,
    BioAuditLog,
    DetectedSpike,
    MEAConfig,
    MultiWellPlate,
    PharmModel,
    WellConfig,
)


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


# ── Multi-Well Plate Tests (Gap 5) ─────────────────────────────────────


class TestMultiWellPlate:
    def test_standard_6_well(self) -> None:
        plate = MultiWellPlate.standard_6_well()
        assert plate.num_wells == 6

    def test_get_well(self) -> None:
        plate = MultiWellPlate.standard_6_well()
        w = plate.get_well("W1")
        assert w is not None
        assert w.well_id == "W1"

    def test_well_label(self) -> None:
        w = WellConfig(
            well_id="W1", mea_config=MEAConfig(), culture_type="hippocampal", passage_number=3
        )
        assert w.label == "W1_hippocampal_P3"

    def test_get_missing_well(self) -> None:
        plate = MultiWellPlate.standard_6_well()
        assert plate.get_well("W99") is None


# ── Network Burst Detection Tests (Gap 6) ─────────────────────────────


class TestBioAuditLog:
    def test_log_entry(self) -> None:
        log = BioAuditLog(experiment_id="EXP001")
        log.log(BioAuditEntry(1, "2026-04-16T08:00:00", 100, 5, 500.0, 0.95))
        assert log.total_rounds == 1

    def test_to_list(self) -> None:
        log = BioAuditLog()
        log.log(BioAuditEntry(1, "2026-04-16", 50, 3, 300.0, 0.9))
        entries = log.to_list()
        assert entries[0]["round"] == 1
        assert entries[0]["spikes"] == 50

    def test_checksum_deterministic(self) -> None:
        log = BioAuditLog()
        log.log(BioAuditEntry(1, "2026-04-16", 50, 3, 300.0, 0.9))
        c1 = log.checksum()
        c2 = log.checksum()
        assert c1 == c2
        assert len(c1) == 64  # SHA-256 hex

    def test_checksum_changes(self) -> None:
        log = BioAuditLog()
        log.log(BioAuditEntry(1, "2026-04-16", 50, 3, 300.0, 0.9))
        c1 = log.checksum()
        log.log(BioAuditEntry(2, "2026-04-16", 60, 4, 400.0, 0.8))
        c2 = log.checksum()
        assert c1 != c2

    def test_checksum_uses_canonical_schema_and_experiment_identity(self) -> None:
        log = BioAuditLog(experiment_id="EXP001")
        log.log(BioAuditEntry(1, "2026-04-16", 50, 3, 300.0, 0.9))
        payload = {
            "schema": "sc-neurocore.bioware-audit.v1",
            "experiment_id": "EXP001",
            "entries": log.to_list(),
        }
        expected = hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()

        assert log.checksum() == expected


# ── Bitstream Rate Decoder Tests (Gap 9) ──────────────────────────────
