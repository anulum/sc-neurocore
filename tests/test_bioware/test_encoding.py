# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bioware encoding tests

"""Tests for AER, stochastic-bitstream, and optical encoding."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.bioware.bioware import (
    AEREvent,
    AERToSCConverter,
    DetectedSpike,
    MEAToAERTranscoder,
    SCToOptoEncoder,
    decode_bitstream_rate,
)


class TestMEAToAERTranscoder:
    def test_transcode(self) -> None:
        spikes = [
            DetectedSpike(channel=0, timestamp_s=0.001, amplitude_uv=-50),
            DetectedSpike(channel=3, timestamp_s=0.005, amplitude_uv=-40),
        ]
        tc = MEAToAERTranscoder(hw_clock_hz=1e6)
        events = tc.transcode(spikes)
        assert len(events) == 2

    def test_timestamp_conversion(self) -> None:
        spikes = [DetectedSpike(channel=0, timestamp_s=0.001, amplitude_uv=-50)]
        tc = MEAToAERTranscoder(hw_clock_hz=1e6)
        events = tc.transcode(spikes)
        assert events[0].timestamp == 1000  # 0.001s * 1MHz = 1000

    def test_channel_mapping(self) -> None:
        spikes = [DetectedSpike(channel=5, timestamp_s=0.0, amplitude_uv=-50)]
        tc = MEAToAERTranscoder(channel_map={5: 42})
        events = tc.transcode(spikes)
        assert events[0].neuron_id == 42

    def test_sorted_by_time(self) -> None:
        spikes = [
            DetectedSpike(channel=0, timestamp_s=0.005, amplitude_uv=-50),
            DetectedSpike(channel=1, timestamp_s=0.001, amplitude_uv=-30),
        ]
        tc = MEAToAERTranscoder()
        events = tc.transcode(spikes)
        assert events[0].timestamp <= events[1].timestamp


# ── AERToSCConverter Tests ───────────────────────────────────────────


class TestAERToSCConverter:
    def test_convert(self) -> None:
        events = [
            AEREvent(neuron_id=0, timestamp=100),
            AEREvent(neuron_id=0, timestamp=200),
            AEREvent(neuron_id=1, timestamp=150),
        ]
        conv = AERToSCConverter(bitstream_length=128)
        bitstreams = conv.convert(events)
        assert 0 in bitstreams
        assert 1 in bitstreams
        assert len(bitstreams[0]) == 128

    def test_density_proportional(self) -> None:
        events = [AEREvent(neuron_id=0, timestamp=i) for i in range(10)]
        events += [AEREvent(neuron_id=1, timestamp=i) for i in range(5)]
        conv = AERToSCConverter(bitstream_length=1024)
        bs = conv.convert(events)
        d0 = float(np.sum(bs[0])) / len(bs[0])
        d1 = float(np.sum(bs[1])) / len(bs[1])
        assert d0 > d1

    def test_empty_events(self) -> None:
        conv = AERToSCConverter()
        bs = conv.convert([])
        assert len(bs) == 0

    def test_lfsr_encode_zero_seed_is_reset(self) -> None:
        # A zero LFSR register is a fixed point; with lfsr_seed=0 and neuron 0
        # the derived seed is 0 and must be bumped to 1 before stepping.
        conv = AERToSCConverter(bitstream_length=64, num_neurons=4, lfsr_seed=0)
        bits = conv._lfsr_encode(0.5, neuron_id=0)
        assert bits.shape == (64,)
        assert bits.dtype == np.uint8


# ── SCToOptoEncoder Tests ────────────────────────────────────────────


class TestSCToOptoEncoder:
    def test_encode(self) -> None:
        bs = {0: np.ones(128, dtype=np.uint8), 1: np.zeros(128, dtype=np.uint8)}
        enc = SCToOptoEncoder(max_intensity_mw_mm2=5.0)
        pulses = enc.encode(bs)
        assert len(pulses) == 1  # neuron 1 is silent, skipped

    def test_intensity_scaling(self) -> None:
        bs = {0: np.ones(100, dtype=np.uint8)}
        enc = SCToOptoEncoder(max_intensity_mw_mm2=10.0)
        pulses = enc.encode(bs)
        assert pulses[0].intensity_mw_mm2 == 10.0

    def test_wavelength(self) -> None:
        bs = {0: np.ones(100, dtype=np.uint8)}
        enc = SCToOptoEncoder(wavelength_nm=590)
        pulses = enc.encode(bs)
        assert pulses[0].wavelength_nm == 590

    def test_duration_range(self) -> None:
        bs = {0: np.ones(100, dtype=np.uint8)}
        enc = SCToOptoEncoder(min_pulse_ms=1.0, max_pulse_ms=50.0)
        pulses = enc.encode(bs)
        assert enc.min_pulse_ms <= pulses[0].duration_ms <= enc.max_pulse_ms


# ── BiologicalSTDP Tests ─────────────────────────────────────────────


class TestOptoSafety:
    def test_power_cap(self) -> None:
        # Create many active neurons exceeding total cap
        bs = {i: np.ones(100, dtype=np.uint8) for i in range(100)}
        enc = SCToOptoEncoder(
            max_intensity_mw_mm2=5.0,
            max_total_power_mw=10.0,
        )
        pulses = enc.encode(bs)
        total_mw = sum(p.power_mw for p in pulses)
        assert total_mw <= 10.0

    def test_no_cap_violation_with_few(self) -> None:
        bs = {0: np.ones(100, dtype=np.uint8)}
        enc = SCToOptoEncoder(max_total_power_mw=50.0)
        pulses = enc.encode(bs)
        assert len(pulses) == 1


# ── Edge Case Tests ──────────────────────────────────────────────────


class TestBitstreamRateDecoder:
    def test_full_density(self) -> None:
        bs = {0: np.ones(256, dtype=np.uint8)}
        rates = decode_bitstream_rate(bs, sc_clock_hz=1e6)
        assert rates[0] == 1e6

    def test_half_density(self) -> None:
        bs_data = np.zeros(256, dtype=np.uint8)
        bs_data[:128] = 1
        rates = decode_bitstream_rate({0: bs_data}, sc_clock_hz=1e6)
        assert rates[0] == pytest.approx(500000.0)

    def test_empty_bitstream(self) -> None:
        rates = decode_bitstream_rate({0: np.array([], dtype=np.uint8)})
        assert rates[0] == 0.0


# ── Homeostatic Plasticity Tests (Gap 10) ──────────────────────────────
