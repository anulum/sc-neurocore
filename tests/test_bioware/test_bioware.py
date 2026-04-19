# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bio-Hybrid Wetware Interface Tests

import sys
import os

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "src", "sc_neurocore", "bioware")
)

from bioware import (
    AEREvent,
    AERToSCConverter,
    ArtifactRejector,
    BCMPlasticity,
    BioAuditEntry,
    BioAuditLog,
    BioHybridSession,
    BiologicalSTDP,
    CultureHealth,
    DetectedSpike,
    HomeostaticPlasticity,
    LatencyBudget,
    LFPBand,
    MEAConfig,
    MEALayout,
    MEAToAERTranscoder,
    MultiWellPlate,
    PharmModel,
    SCToOptoEncoder,
    SpikeDetector,
    SpikeSorter,
    WellConfig,
    decode_bitstream_rate,
    detect_network_bursts,
    extract_lfp_power,
)


# ── helpers ──────────────────────────────────────────────────────────


def _synth_voltage(n_samples: int = 2000, n_channels: int = 10, seed: int = 42) -> np.ndarray:
    """Generate synthetic MEA voltage data with embedded spikes."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0, 5, size=(n_samples, n_channels))
    # Inject clear spikes on channel 0 and 3
    for i in range(0, n_samples, 200):
        if i < n_samples:
            data[i, 0] = -80.0
        if i + 50 < n_samples:
            data[i + 50, 3] = -60.0
    return data


# ── MEAConfig Tests ──────────────────────────────────────────────────


class TestMEAConfig:
    def test_defaults(self):
        cfg = MEAConfig()
        assert cfg.num_channels == 60
        assert cfg.sample_rate_hz == 20_000.0

    def test_from_layout_60(self):
        cfg = MEAConfig.from_layout(MEALayout.MEA_60)
        assert cfg.num_channels == 60

    def test_from_layout_4096(self):
        cfg = MEAConfig.from_layout(MEALayout.MEA_4096)
        assert cfg.num_channels == 4096
        assert cfg.electrode_pitch_um < 20.0

    def test_all_layouts(self):
        for layout in MEALayout:
            cfg = MEAConfig.from_layout(layout)
            assert cfg.num_channels > 0


# ── SpikeDetector Tests ──────────────────────────────────────────────


class TestSpikeDetector:
    def test_estimate_noise(self):
        cfg = MEAConfig(num_channels=10)
        det = SpikeDetector(config=cfg)
        data = _synth_voltage()
        noise = det.estimate_noise(data)
        assert len(noise) == 10
        assert np.all(noise > 0)

    def test_detect_spikes(self):
        cfg = MEAConfig(num_channels=10, spike_threshold_sigma=3.0)
        det = SpikeDetector(config=cfg)
        data = _synth_voltage()
        spikes = det.detect(data)
        assert len(spikes) > 0

    def test_spike_channels(self):
        cfg = MEAConfig(num_channels=10, spike_threshold_sigma=3.0)
        det = SpikeDetector(config=cfg)
        data = _synth_voltage()
        spikes = det.detect(data)
        channels = set(s.channel for s in spikes)
        assert 0 in channels  # We injected spikes on channel 0

    def test_spike_has_timestamp(self):
        cfg = MEAConfig(num_channels=10)
        det = SpikeDetector(config=cfg)
        spikes = det.detect(_synth_voltage())
        for s in spikes:
            assert s.timestamp_s >= 0


# ── MEAToAERTranscoder Tests ─────────────────────────────────────────


class TestMEAToAERTranscoder:
    def test_transcode(self):
        spikes = [
            DetectedSpike(channel=0, timestamp_s=0.001, amplitude_uv=-50),
            DetectedSpike(channel=3, timestamp_s=0.005, amplitude_uv=-40),
        ]
        tc = MEAToAERTranscoder(hw_clock_hz=1e6)
        events = tc.transcode(spikes)
        assert len(events) == 2

    def test_timestamp_conversion(self):
        spikes = [DetectedSpike(channel=0, timestamp_s=0.001, amplitude_uv=-50)]
        tc = MEAToAERTranscoder(hw_clock_hz=1e6)
        events = tc.transcode(spikes)
        assert events[0].timestamp == 1000  # 0.001s * 1MHz = 1000

    def test_channel_mapping(self):
        spikes = [DetectedSpike(channel=5, timestamp_s=0.0, amplitude_uv=-50)]
        tc = MEAToAERTranscoder(channel_map={5: 42})
        events = tc.transcode(spikes)
        assert events[0].neuron_id == 42

    def test_sorted_by_time(self):
        spikes = [
            DetectedSpike(channel=0, timestamp_s=0.005, amplitude_uv=-50),
            DetectedSpike(channel=1, timestamp_s=0.001, amplitude_uv=-30),
        ]
        tc = MEAToAERTranscoder()
        events = tc.transcode(spikes)
        assert events[0].timestamp <= events[1].timestamp


# ── AERToSCConverter Tests ───────────────────────────────────────────


class TestAERToSCConverter:
    def test_convert(self):
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

    def test_density_proportional(self):
        events = [AEREvent(neuron_id=0, timestamp=i) for i in range(10)]
        events += [AEREvent(neuron_id=1, timestamp=i) for i in range(5)]
        conv = AERToSCConverter(bitstream_length=1024)
        bs = conv.convert(events)
        d0 = float(np.sum(bs[0])) / len(bs[0])
        d1 = float(np.sum(bs[1])) / len(bs[1])
        assert d0 > d1

    def test_empty_events(self):
        conv = AERToSCConverter()
        bs = conv.convert([])
        assert len(bs) == 0


# ── SCToOptoEncoder Tests ────────────────────────────────────────────


class TestSCToOptoEncoder:
    def test_encode(self):
        bs = {0: np.ones(128, dtype=np.uint8), 1: np.zeros(128, dtype=np.uint8)}
        enc = SCToOptoEncoder(max_intensity_mw_mm2=5.0)
        pulses = enc.encode(bs)
        assert len(pulses) == 1  # neuron 1 is silent, skipped

    def test_intensity_scaling(self):
        bs = {0: np.ones(100, dtype=np.uint8)}
        enc = SCToOptoEncoder(max_intensity_mw_mm2=10.0)
        pulses = enc.encode(bs)
        assert pulses[0].intensity_mw_mm2 == 10.0

    def test_wavelength(self):
        bs = {0: np.ones(100, dtype=np.uint8)}
        enc = SCToOptoEncoder(wavelength_nm=590)
        pulses = enc.encode(bs)
        assert pulses[0].wavelength_nm == 590

    def test_duration_range(self):
        bs = {0: np.ones(100, dtype=np.uint8)}
        enc = SCToOptoEncoder(min_pulse_ms=1.0, max_pulse_ms=50.0)
        pulses = enc.encode(bs)
        assert enc.min_pulse_ms <= pulses[0].duration_ms <= enc.max_pulse_ms


# ── BiologicalSTDP Tests ─────────────────────────────────────────────


class TestBiologicalSTDP:
    def test_potentiation(self):
        stdp = BiologicalSTDP()
        dw = stdp.compute_dw(5.0)  # post after pre
        assert dw > 0

    def test_depression(self):
        stdp = BiologicalSTDP()
        dw = stdp.compute_dw(-5.0)  # pre after post
        assert dw < 0

    def test_zero_dt(self):
        stdp = BiologicalSTDP()
        assert stdp.compute_dw(0.0) == 0.0

    def test_exponential_decay(self):
        stdp = BiologicalSTDP(tau_plus_ms=20.0)
        dw_near = stdp.compute_dw(1.0)
        dw_far = stdp.compute_dw(40.0)
        assert abs(dw_near) > abs(dw_far)

    def test_update_weight_bounded(self):
        stdp = BiologicalSTDP(w_max_q88=512, w_min_q88=0)
        w = stdp.update_weight(500, 1.0)
        assert w <= 512
        w = stdp.update_weight(5, -100.0)
        assert w >= 0


# ── BCMPlasticity Tests ──────────────────────────────────────────────


class TestBCMPlasticity:
    def test_threshold_update(self):
        bcm = BCMPlasticity()
        bcm.update_theta(10.0, dt_ms=10.0)
        assert bcm.theta > 0

    def test_ltp_above_threshold(self):
        bcm = BCMPlasticity()
        bcm.theta = 5.0
        dw = bcm.compute_dw(10.0, 10.0)  # post > theta
        assert dw > 0

    def test_ltd_below_threshold(self):
        bcm = BCMPlasticity()
        bcm.theta = 20.0
        dw = bcm.compute_dw(10.0, 10.0)  # post < theta
        assert dw < 0

    def test_weight_bounded(self):
        bcm = BCMPlasticity(w_max_q88=512, w_min_q88=0)
        bcm.theta = 0.0
        w = bcm.update_weight(510, 100.0, 100.0)
        assert w <= 512


# ── CultureHealth Tests ─────────────────────────────────────────────


class TestCultureHealth:
    def test_healthy_culture(self):
        ch = CultureHealth(min_active_channels=3)
        counts = np.array([10, 15, 20, 5, 8, 0, 0, 0])
        result = ch.assess(counts, duration_s=1.0)
        assert result["is_viable"] is True

    def test_dead_culture(self):
        ch = CultureHealth(min_active_channels=5)
        counts = np.zeros(60)
        result = ch.assess(counts, duration_s=1.0)
        assert result["health_score"] == 0.0
        assert result["is_viable"] is False

    def test_bursting_detection(self):
        ch = CultureHealth(burst_threshold_hz=50.0)
        counts = np.array([100, 200, 5, 3])
        result = ch.assess(counts, duration_s=1.0)
        assert result["bursting_channels"] == 2


# ── BioHybridSession Tests ───────────────────────────────────────────


class TestBioHybridSession:
    def _make_session(self):
        cfg = MEAConfig(num_channels=10)
        det = SpikeDetector(config=cfg)
        tc = MEAToAERTranscoder(hw_clock_hz=1e6)
        sc = AERToSCConverter(bitstream_length=128, num_neurons=10)
        opto = SCToOptoEncoder()
        return BioHybridSession(
            mea_config=cfg,
            detector=det,
            transcoder=tc,
            sc_converter=sc,
            opto_encoder=opto,
        )

    def test_process_frame(self):
        session = self._make_session()
        data = _synth_voltage(n_channels=10)
        result = session.process_frame(data)
        assert result["round"] == 1
        assert result["num_spikes"] > 0

    def test_full_pipeline(self):
        session = self._make_session()
        data = _synth_voltage(n_channels=10)
        result = session.process_frame(data)
        assert "aer_events" in result
        assert "bitstreams" in result
        assert "opto_pulses" in result
        assert "health" in result

    def test_multiple_rounds(self):
        session = self._make_session()
        for i in range(3):
            data = _synth_voltage(n_channels=10, seed=42 + i)
            session.process_frame(data)
        assert session.round_count == 3

    def test_health_in_result(self):
        session = self._make_session()
        data = _synth_voltage(n_channels=10)
        result = session.process_frame(data)
        assert "health_score" in result["health"]
        assert "is_viable" in result["health"]

    def test_latency_measured(self):
        session = self._make_session()
        data = _synth_voltage(n_channels=10)
        result = session.process_frame(data)
        assert "latency_us" in result
        assert result["latency_us"] > 0


# ── Refractory Period Tests ──────────────────────────────────────────


class TestRefractoryPeriod:
    def test_refractory_reduces_spikes(self):
        cfg = MEAConfig(num_channels=10, spike_threshold_sigma=3.0)
        det_no_ref = SpikeDetector(config=cfg, refractory_samples=0)
        det_with_ref = SpikeDetector(config=cfg, refractory_samples=50)
        data = _synth_voltage()
        spikes_no = det_no_ref.detect(data)
        spikes_yes = det_with_ref.detect(data)
        assert len(spikes_yes) <= len(spikes_no)

    def test_refractory_zero_matches_original(self):
        cfg = MEAConfig(num_channels=10, spike_threshold_sigma=3.0)
        # refractory_samples=0 means no refractory filter
        det = SpikeDetector(config=cfg, refractory_samples=0)
        data = _synth_voltage()
        spikes = det.detect(data)
        assert len(spikes) > 0


# ── Optogenetic Safety Tests ─────────────────────────────────────────


class TestOptoSafety:
    def test_power_cap(self):
        # Create many active neurons exceeding total cap
        bs = {i: np.ones(100, dtype=np.uint8) for i in range(100)}
        enc = SCToOptoEncoder(
            max_intensity_mw_mm2=5.0,
            max_total_power_mw=10.0,
        )
        pulses = enc.encode(bs)
        total_mw = sum(p.intensity_mw_mm2 for p in pulses)
        assert total_mw <= 10.0 + 5.0  # at most one pulse over

    def test_no_cap_violation_with_few(self):
        bs = {0: np.ones(100, dtype=np.uint8)}
        enc = SCToOptoEncoder(max_total_power_mw=50.0)
        pulses = enc.encode(bs)
        assert len(pulses) == 1


# ── Edge Case Tests ──────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_sample_data(self):
        cfg = MEAConfig(num_channels=5)
        det = SpikeDetector(config=cfg)
        data = np.zeros((1, 5))
        spikes = det.detect(data)
        assert len(spikes) == 0

    def test_all_silent_neurons_no_pulses(self):
        bs = {0: np.zeros(100, dtype=np.uint8)}
        enc = SCToOptoEncoder()
        pulses = enc.encode(bs)
        assert len(pulses) == 0

    def test_stdp_symmetry(self):
        stdp = BiologicalSTDP(tau_plus_ms=20.0, tau_minus_ms=20.0, a_plus=0.01, a_minus=0.01)
        dw_pos = stdp.compute_dw(5.0)
        dw_neg = stdp.compute_dw(-5.0)
        assert abs(dw_pos + dw_neg) < 1e-10

    def test_bcm_zero_rate(self):
        bcm = BCMPlasticity()
        dw = bcm.compute_dw(0.0, 0.0)
        assert dw == 0.0

    def test_culture_health_zero_duration(self):
        ch = CultureHealth()
        counts = np.array([5, 10])
        result = ch.assess(counts, duration_s=0.0)
        assert "health_score" in result

    def test_aer_to_sc_invalid_event(self):
        events = [AEREvent(neuron_id=0, timestamp=100, valid=False)]
        conv = AERToSCConverter()
        bs = conv.convert(events)
        assert len(bs) == 0


# ── Spike Sorter Tests (Gap 1) ─────────────────────────────────────────


class TestSpikeSorter:
    def test_fit_and_assign(self):
        spikes = [
            DetectedSpike(channel=0, timestamp_s=0.0, amplitude_uv=-30),
            DetectedSpike(channel=0, timestamp_s=0.01, amplitude_uv=-60),
            DetectedSpike(channel=0, timestamp_s=0.02, amplitude_uv=-90),
        ]
        sorter = SpikeSorter(num_units=3)
        sorter.fit(spikes)
        sorted_spikes = sorter.assign(spikes)
        assert len(sorted_spikes) == 3
        units = set(s.unit_id for s in sorted_spikes)
        assert len(units) > 1

    def test_no_fit_returns_original(self):
        sorter = SpikeSorter(num_units=3)
        spikes = [DetectedSpike(channel=0, timestamp_s=0.0, amplitude_uv=-50)]
        result = sorter.assign(spikes)
        assert result[0].unit_id == 0

    def test_empty_spikes(self):
        sorter = SpikeSorter()
        sorter.fit([])
        result = sorter.assign([])
        assert len(result) == 0


# ── LFP Extraction Tests (Gap 2) ───────────────────────────────────────


class TestLFPExtraction:
    def test_default_bands(self):
        data = _synth_voltage(n_samples=2000, n_channels=5)
        result = extract_lfp_power(data, sample_rate_hz=20000.0)
        assert "delta" in result
        assert "gamma" in result
        assert result["delta"].shape == (5,)

    def test_custom_band(self):
        data = _synth_voltage(n_samples=2000, n_channels=5)
        bands = [LFPBand("custom", 10.0, 50.0)]
        result = extract_lfp_power(data, sample_rate_hz=20000.0, bands=bands)
        assert "custom" in result
        assert np.all(result["custom"] >= 0)


# ── Latency Budget Tests (Gap 3) ───────────────────────────────────────


class TestLatencyBudget:
    def test_within_budget(self):
        lb = LatencyBudget(max_latency_us=1000.0)
        assert lb.record(500.0) is True
        assert lb.violations == 0

    def test_exceeds_budget(self):
        lb = LatencyBudget(max_latency_us=1000.0)
        assert lb.record(1500.0) is False
        assert lb.violations == 1

    def test_compliance_ratio(self):
        lb = LatencyBudget(max_latency_us=1000.0)
        lb.record(500.0)
        lb.record(500.0)
        lb.record(1500.0)
        assert lb.compliance_ratio == pytest.approx(2.0 / 3.0)

    def test_p99_latency(self):
        lb = LatencyBudget()
        for i in range(100):
            lb.record(float(i))
        assert lb.p99_latency_us > 90.0


# ── PharmModel Tests (Gap 4) ───────────────────────────────────────────


class TestPharmModel:
    def test_no_drug(self):
        pm = PharmModel()
        assert pm.effective_gain(0.0) == 1.0

    def test_full_onset(self):
        pm = PharmModel(gain=2.0, onset_delay_s=10.0)
        pm.apply(0.0)
        assert pm.effective_gain(100.0) == 2.0

    def test_partial_onset(self):
        pm = PharmModel(gain=2.0, onset_delay_s=10.0)
        pm.apply(0.0)
        g = pm.effective_gain(5.0)  # half onset
        assert 1.0 < g < 2.0

    def test_modulate_spikes(self):
        pm = PharmModel(gain=0.0, onset_delay_s=0.0)  # TTX silencing
        pm.apply(0.0)
        counts = np.array([10, 20, 30])
        result = pm.modulate_spikes(counts, 100.0)
        assert np.all(result == 0)


# ── Multi-Well Plate Tests (Gap 5) ─────────────────────────────────────


class TestMultiWellPlate:
    def test_standard_6_well(self):
        plate = MultiWellPlate.standard_6_well()
        assert plate.num_wells == 6

    def test_get_well(self):
        plate = MultiWellPlate.standard_6_well()
        w = plate.get_well("W1")
        assert w is not None
        assert w.well_id == "W1"

    def test_well_label(self):
        w = WellConfig(
            well_id="W1", mea_config=MEAConfig(), culture_type="hippocampal", passage_number=3
        )
        assert w.label == "W1_hippocampal_P3"

    def test_get_missing_well(self):
        plate = MultiWellPlate.standard_6_well()
        assert plate.get_well("W99") is None


# ── Network Burst Detection Tests (Gap 6) ─────────────────────────────


class TestNetworkBurstDetection:
    def test_synchronised_burst(self):
        rng = np.random.default_rng(42)
        spikes = []
        # Background: sparse spikes
        for i in range(100):
            spikes.append(
                DetectedSpike(
                    channel=rng.integers(0, 10), timestamp_s=rng.uniform(0, 1), amplitude_uv=-30
                )
            )
        # Burst: many spikes from many channels at t=0.5
        for ch in range(8):
            for _ in range(10):
                spikes.append(
                    DetectedSpike(
                        channel=ch, timestamp_s=0.5 + rng.uniform(-0.005, 0.005), amplitude_uv=-50
                    )
                )
        bursts = detect_network_bursts(
            spikes, bin_width_s=0.01, threshold_sigma=2.0, min_channels=5
        )
        assert len(bursts) > 0
        assert bursts[0].participating_channels >= 5

    def test_no_burst(self):
        spikes = [
            DetectedSpike(channel=0, timestamp_s=float(i), amplitude_uv=-30) for i in range(10)
        ]
        bursts = detect_network_bursts(spikes, min_channels=3)
        assert len(bursts) == 0

    def test_empty_spikes(self):
        assert detect_network_bursts([]) == []


# ── Artifact Rejection Tests (Gap 7) ───────────────────────────────────


class TestArtifactRejection:
    def test_blanking(self):
        data = np.ones((1000, 5))
        ar = ArtifactRejector(blanking_pre_ms=0.5, blanking_post_ms=2.0)
        blanked = ar.blank(data, stim_times_s=[0.025], sample_rate_hz=20000.0)
        # Centre at sample 500, pre=10 post=40 → blanked
        assert blanked[500, 0] == 0.0

    def test_no_stim_no_blanking(self):
        data = np.ones((100, 3))
        ar = ArtifactRejector()
        blanked = ar.blank(data, stim_times_s=[], sample_rate_hz=20000.0)
        np.testing.assert_array_equal(blanked, data)


# ── Bio Audit Log Tests (Gap 8) ────────────────────────────────────────


class TestBioAuditLog:
    def test_log_entry(self):
        log = BioAuditLog(experiment_id="EXP001")
        log.log(BioAuditEntry(1, "2026-04-16T08:00:00", 100, 5, 500.0, 0.95))
        assert log.total_rounds == 1

    def test_to_list(self):
        log = BioAuditLog()
        log.log(BioAuditEntry(1, "2026-04-16", 50, 3, 300.0, 0.9))
        entries = log.to_list()
        assert entries[0]["round"] == 1
        assert entries[0]["spikes"] == 50

    def test_checksum_deterministic(self):
        log = BioAuditLog()
        log.log(BioAuditEntry(1, "2026-04-16", 50, 3, 300.0, 0.9))
        c1 = log.checksum()
        c2 = log.checksum()
        assert c1 == c2
        assert len(c1) == 64  # SHA-256 hex

    def test_checksum_changes(self):
        log = BioAuditLog()
        log.log(BioAuditEntry(1, "2026-04-16", 50, 3, 300.0, 0.9))
        c1 = log.checksum()
        log.log(BioAuditEntry(2, "2026-04-16", 60, 4, 400.0, 0.8))
        c2 = log.checksum()
        assert c1 != c2


# ── Bitstream Rate Decoder Tests (Gap 9) ──────────────────────────────


class TestBitstreamRateDecoder:
    def test_full_density(self):
        bs = {0: np.ones(256, dtype=np.uint8)}
        rates = decode_bitstream_rate(bs, sc_clock_hz=1e6)
        assert rates[0] == 1e6

    def test_half_density(self):
        bs_data = np.zeros(256, dtype=np.uint8)
        bs_data[:128] = 1
        rates = decode_bitstream_rate({0: bs_data}, sc_clock_hz=1e6)
        assert rates[0] == pytest.approx(500000.0)

    def test_empty_bitstream(self):
        rates = decode_bitstream_rate({0: np.array([], dtype=np.uint8)})
        assert rates[0] == 0.0


# ── Homeostatic Plasticity Tests (Gap 10) ──────────────────────────────


class TestHomeostaticPlasticity:
    def test_at_target_no_change(self):
        hp = HomeostaticPlasticity(target_rate_hz=10.0)
        new = hp.update_threshold(256, observed_rate_hz=10.0, dt_ms=100.0)
        assert new == 256

    def test_too_fast_increases_threshold(self):
        hp = HomeostaticPlasticity(target_rate_hz=10.0, tau_homeo_ms=1000.0)
        new = hp.update_threshold(256, observed_rate_hz=50.0, dt_ms=1000.0)
        assert new > 256

    def test_too_slow_decreases_threshold(self):
        hp = HomeostaticPlasticity(target_rate_hz=10.0, tau_homeo_ms=1000.0)
        new = hp.update_threshold(256, observed_rate_hz=1.0, dt_ms=1000.0)
        assert new < 256

    def test_bounded(self):
        hp = HomeostaticPlasticity(max_threshold_q88=512, min_threshold_q88=64)
        new = hp.update_threshold(500, observed_rate_hz=1000.0, dt_ms=10000.0)
        assert new <= 512
        new = hp.update_threshold(70, observed_rate_hz=0.0, dt_ms=10000.0)
        assert new >= 64
