# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bio-Hybrid Wetware Interface Tests

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import MutableMapping
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.bioware.bioware import (
    AEREvent,
    AERToSCConverter,
    ArtifactRejector,
    BCMPlasticity,
    BioAuditEntry,
    BioAuditLog,
    BioHybridFrameResult,
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
    mea_fitness_hook,
)


FloatArray = npt.NDArray[np.float64]


# ── helpers ──────────────────────────────────────────────────────────


def _synth_voltage(n_samples: int = 2000, n_channels: int = 10, seed: int = 42) -> FloatArray:
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
    def test_defaults(self) -> None:
        cfg = MEAConfig()
        assert cfg.num_channels == 60
        assert cfg.sample_rate_hz == 20_000.0

    def test_from_layout_60(self) -> None:
        cfg = MEAConfig.from_layout(MEALayout.MEA_60)
        assert cfg.num_channels == 60

    def test_from_layout_4096(self) -> None:
        cfg = MEAConfig.from_layout(MEALayout.MEA_4096)
        assert cfg.num_channels == 4096
        assert cfg.electrode_pitch_um < 20.0

    def test_all_layouts(self) -> None:
        for layout in MEALayout:
            cfg = MEAConfig.from_layout(layout)
            assert cfg.num_channels > 0


# ── SpikeDetector Tests ──────────────────────────────────────────────


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


class TestBiologicalSTDP:
    def test_potentiation(self) -> None:
        stdp = BiologicalSTDP()
        dw = stdp.compute_dw(5.0)  # post after pre
        assert dw > 0

    def test_depression(self) -> None:
        stdp = BiologicalSTDP()
        dw = stdp.compute_dw(-5.0)  # pre after post
        assert dw < 0

    def test_zero_dt(self) -> None:
        stdp = BiologicalSTDP()
        assert stdp.compute_dw(0.0) == 0.0

    def test_exponential_decay(self) -> None:
        stdp = BiologicalSTDP(tau_plus_ms=20.0)
        dw_near = stdp.compute_dw(1.0)
        dw_far = stdp.compute_dw(40.0)
        assert abs(dw_near) > abs(dw_far)

    def test_update_weight_bounded(self) -> None:
        stdp = BiologicalSTDP(w_max_q88=512, w_min_q88=0)
        w = stdp.update_weight(500, 1.0)
        assert w <= 512
        w = stdp.update_weight(5, -100.0)
        assert w >= 0


# ── BCMPlasticity Tests ──────────────────────────────────────────────


class TestBCMPlasticity:
    def test_threshold_update(self) -> None:
        bcm = BCMPlasticity()
        bcm.update_theta(10.0, dt_ms=10.0)
        assert bcm.theta > 0

    def test_ltp_above_threshold(self) -> None:
        bcm = BCMPlasticity()
        bcm.theta = 5.0
        dw = bcm.compute_dw(10.0, 10.0)  # post > theta
        assert dw > 0

    def test_ltd_below_threshold(self) -> None:
        bcm = BCMPlasticity()
        bcm.theta = 20.0
        dw = bcm.compute_dw(10.0, 10.0)  # post < theta
        assert dw < 0

    def test_weight_bounded(self) -> None:
        bcm = BCMPlasticity(w_max_q88=512, w_min_q88=0)
        bcm.theta = 0.0
        w = bcm.update_weight(510, 100.0, 100.0)
        assert w <= 512


# ── CultureHealth Tests ─────────────────────────────────────────────


class TestCultureHealth:
    def test_healthy_culture(self) -> None:
        ch = CultureHealth(min_active_channels=3)
        counts = np.array([10, 15, 20, 5, 8, 0, 0, 0])
        result = ch.assess(counts, duration_s=1.0)
        assert result["is_viable"] is True

    def test_dead_culture(self) -> None:
        ch = CultureHealth(min_active_channels=5)
        counts = np.zeros(60)
        result = ch.assess(counts, duration_s=1.0)
        assert result["health_score"] == 0.0
        assert result["is_viable"] is False

    def test_bursting_detection(self) -> None:
        ch = CultureHealth(burst_threshold_hz=50.0)
        counts = np.array([100, 200, 5, 3])
        result = ch.assess(counts, duration_s=1.0)
        assert result["bursting_channels"] == 2

    def test_excessive_firing_rate_caps_health(self) -> None:
        # A mean rate above the hyperactivity ceiling scales the health score
        # down rather than leaving it at 1.0.
        ch = CultureHealth(min_active_channels=1, max_firing_rate_hz=10.0)
        counts = np.full(8, 1000.0)
        result = ch.assess(counts, duration_s=1.0)
        assert result["health_score"] < 1.0


# ── BioHybridSession Tests ───────────────────────────────────────────


class TestBioHybridSession:
    def _make_session(self) -> BioHybridSession:
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

    def test_process_frame(self) -> None:
        session = self._make_session()
        data = _synth_voltage(n_channels=10)
        result = session.process_frame(data)
        assert result["round"] == 1
        assert result["num_spikes"] > 0

    def test_full_pipeline(self) -> None:
        session = self._make_session()
        data = _synth_voltage(n_channels=10)
        result = session.process_frame(data)
        assert "aer_events" in result
        assert "bitstreams" in result
        assert "opto_pulses" in result
        assert "health" in result

    def test_multiple_rounds(self) -> None:
        session = self._make_session()
        for i in range(3):
            data = _synth_voltage(n_channels=10, seed=42 + i)
            session.process_frame(data)
        assert session.round_count == 3

    def test_health_in_result(self) -> None:
        session = self._make_session()
        data = _synth_voltage(n_channels=10)
        result = session.process_frame(data)
        assert "health_score" in result["health"]
        assert "is_viable" in result["health"]

    def test_latency_measured(self) -> None:
        session = self._make_session()
        data = _synth_voltage(n_channels=10)
        result = session.process_frame(data)
        assert "latency_us" in result
        assert result["latency_us"] > 0

    def test_process_frame_runs_all_optional_stages(self) -> None:
        cfg = MEAConfig(num_channels=10)
        captured: dict[str, object] = {}

        class _ZenithStub:
            def step_from_bio_rates(self, rates: dict[int, float]) -> None:
                captured["rates"] = rates

        session = BioHybridSession(
            mea_config=cfg,
            detector=SpikeDetector(config=cfg),
            transcoder=MEAToAERTranscoder(hw_clock_hz=1e6),
            sc_converter=AERToSCConverter(bitstream_length=128, num_neurons=10),
            opto_encoder=SCToOptoEncoder(),
            artifact_rejector=ArtifactRejector(),
            sorter=SpikeSorter(num_units=3),
            pharm_model=PharmModel(),
            latency_budget=LatencyBudget(),
            zenith_core=cast(Any, _ZenithStub()),
        )
        data = _synth_voltage(n_channels=10)
        result = session.process_frame(data, stim_times_s=[0.001])
        assert result["round"] == 1
        assert "rates" in captured  # the zenith stage received decoded rates


# ── Refractory Period Tests ──────────────────────────────────────────


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


class TestOptoSafety:
    def test_power_cap(self) -> None:
        # Create many active neurons exceeding total cap
        bs = {i: np.ones(100, dtype=np.uint8) for i in range(100)}
        enc = SCToOptoEncoder(
            max_intensity_mw_mm2=5.0,
            max_total_power_mw=10.0,
        )
        pulses = enc.encode(bs)
        total_mw = sum(p.intensity_mw_mm2 for p in pulses)
        assert total_mw <= 10.0 + 5.0  # at most one pulse over

    def test_no_cap_violation_with_few(self) -> None:
        bs = {0: np.ones(100, dtype=np.uint8)}
        enc = SCToOptoEncoder(max_total_power_mw=50.0)
        pulses = enc.encode(bs)
        assert len(pulses) == 1


# ── Edge Case Tests ──────────────────────────────────────────────────


class TestEdgeCases:
    def test_single_sample_data(self) -> None:
        cfg = MEAConfig(num_channels=5)
        det = SpikeDetector(config=cfg)
        data = np.zeros((1, 5))
        spikes = det.detect(data)
        assert len(spikes) == 0

    def test_all_silent_neurons_no_pulses(self) -> None:
        bs = {0: np.zeros(100, dtype=np.uint8)}
        enc = SCToOptoEncoder()
        pulses = enc.encode(bs)
        assert len(pulses) == 0

    def test_stdp_symmetry(self) -> None:
        stdp = BiologicalSTDP(tau_plus_ms=20.0, tau_minus_ms=20.0, a_plus=0.01, a_minus=0.01)
        dw_pos = stdp.compute_dw(5.0)
        dw_neg = stdp.compute_dw(-5.0)
        assert abs(dw_pos + dw_neg) < 1e-10

    def test_bcm_zero_rate(self) -> None:
        bcm = BCMPlasticity()
        dw = bcm.compute_dw(0.0, 0.0)
        assert dw == 0.0

    def test_culture_health_zero_duration(self) -> None:
        ch = CultureHealth()
        counts = np.array([5, 10])
        result = ch.assess(counts, duration_s=0.0)
        assert "health_score" in result

    def test_aer_to_sc_invalid_event(self) -> None:
        events = [AEREvent(neuron_id=0, timestamp=100, valid=False)]
        conv = AERToSCConverter()
        bs = conv.convert(events)
        assert len(bs) == 0


# ── Spike Sorter Tests (Gap 1) ─────────────────────────────────────────


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


class TestLFPExtraction:
    def test_default_bands(self) -> None:
        data = _synth_voltage(n_samples=2000, n_channels=5)
        result = extract_lfp_power(data, sample_rate_hz=20000.0)
        assert "delta" in result
        assert "gamma" in result
        assert result["delta"].shape == (5,)

    def test_custom_band(self) -> None:
        data = _synth_voltage(n_samples=2000, n_channels=5)
        bands = [LFPBand("custom", 10.0, 50.0)]
        result = extract_lfp_power(data, sample_rate_hz=20000.0, bands=bands)
        assert "custom" in result
        assert np.all(result["custom"] >= 0)


# ── Latency Budget Tests (Gap 3) ───────────────────────────────────────


class TestLatencyBudget:
    def test_within_budget(self) -> None:
        lb = LatencyBudget(max_latency_us=1000.0)
        assert lb.record(500.0) is True
        assert lb.violations == 0

    def test_exceeds_budget(self) -> None:
        lb = LatencyBudget(max_latency_us=1000.0)
        assert lb.record(1500.0) is False
        assert lb.violations == 1

    def test_compliance_ratio(self) -> None:
        lb = LatencyBudget(max_latency_us=1000.0)
        lb.record(500.0)
        lb.record(500.0)
        lb.record(1500.0)
        assert lb.compliance_ratio == pytest.approx(2.0 / 3.0)

    def test_p99_latency(self) -> None:
        lb = LatencyBudget()
        for i in range(100):
            lb.record(float(i))
        assert lb.p99_latency_us > 90.0

    def test_mean_latency(self) -> None:
        lb = LatencyBudget()
        lb.record(100.0)
        lb.record(300.0)
        assert lb.mean_latency_us == pytest.approx(200.0)

    def test_compliance_ratio_empty_history(self) -> None:
        assert LatencyBudget().compliance_ratio == 1.0


# ── PharmModel Tests (Gap 4) ───────────────────────────────────────────


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
        pm = PharmModel(gain=-1.0, onset_delay_s=0.0)
        pm.apply(0.0)
        spikes = [DetectedSpike(channel=0, timestamp_s=0.0, amplitude_uv=-40.0)]
        with pytest.raises(ValueError, match="finite and >= 0"):
            pm.modulate_spike_events(spikes, 1.0)

    def test_modulate_unit_gain_preserves_events(self) -> None:
        pm = PharmModel(gain=1.0, onset_delay_s=0.0)
        pm.apply(0.0)
        spikes = [
            DetectedSpike(channel=0, timestamp_s=i * 0.001, amplitude_uv=-40.0) for i in range(4)
        ]
        result = pm.modulate_spike_events(spikes, 1.0)
        assert len(result) == 4  # gain 1.0 -> target count equals input count

    def test_modulate_excitatory_non_finite_timestamp_raises(self) -> None:
        pm = PharmModel(gain=2.0, onset_delay_s=0.0)
        pm.apply(0.0)
        spikes = [
            DetectedSpike(channel=0, timestamp_s=0.0, amplitude_uv=-40.0),
            DetectedSpike(channel=0, timestamp_s=float("inf"), amplitude_uv=-40.0),
        ]
        with pytest.raises(ValueError, match="timestamps must be finite"):
            pm.modulate_spike_events(spikes, 1.0)

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


class TestNetworkBurstDetection:
    def test_synchronised_burst(self) -> None:
        rng = np.random.default_rng(42)
        spikes: list[DetectedSpike] = []
        # Background: sparse spikes
        for i in range(100):
            spikes.append(
                DetectedSpike(
                    channel=int(rng.integers(0, 10)),
                    timestamp_s=float(rng.uniform(0, 1)),
                    amplitude_uv=-30,
                )
            )
        # Burst: many spikes from many channels at t=0.5
        for ch in range(8):
            for _ in range(10):
                spikes.append(
                    DetectedSpike(
                        channel=ch,
                        timestamp_s=0.5 + float(rng.uniform(-0.005, 0.005)),
                        amplitude_uv=-50,
                    )
                )
        bursts = detect_network_bursts(
            spikes, bin_width_s=0.01, threshold_sigma=2.0, min_channels=5
        )
        assert len(bursts) > 0
        assert bursts[0].participating_channels >= 5

    def test_no_burst(self) -> None:
        spikes = [
            DetectedSpike(channel=0, timestamp_s=float(i), amplitude_uv=-30) for i in range(10)
        ]
        bursts = detect_network_bursts(spikes, min_channels=3)
        assert len(bursts) == 0

    def test_empty_spikes(self) -> None:
        assert detect_network_bursts([]) == []

    def test_same_timestamp_spikes_have_no_temporal_span(self) -> None:
        spikes = [
            DetectedSpike(channel=ch, timestamp_s=0.25, amplitude_uv=-40.0) for ch in range(4)
        ]

        assert detect_network_bursts(spikes, min_channels=1) == []

    def test_uniform_bin_counts_have_no_burst_threshold(self) -> None:
        spikes = [
            DetectedSpike(channel=0, timestamp_s=0.00, amplitude_uv=-40.0),
            DetectedSpike(channel=1, timestamp_s=0.01, amplitude_uv=-40.0),
            DetectedSpike(channel=2, timestamp_s=0.02, amplitude_uv=-40.0),
        ]

        assert detect_network_bursts(spikes, bin_width_s=0.01, min_channels=1) == []


# ── Artifact Rejection Tests (Gap 7) ───────────────────────────────────


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

    def test_checksum_falls_back_to_stdlib_json_without_orjson(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        module_cache = cast(MutableMapping[str, object], sys.modules)
        monkeypatch.setitem(module_cache, "orjson", None)
        log = BioAuditLog()
        log.log(BioAuditEntry(1, "2026-04-16", 50, 3, 300.0, 0.9))
        expected = hashlib.sha256(
            json.dumps(log.to_list(), sort_keys=True).encode("utf-8")
        ).hexdigest()

        assert log.checksum() == expected


# ── Bitstream Rate Decoder Tests (Gap 9) ──────────────────────────────


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


class TestHomeostaticPlasticity:
    def test_at_target_no_change(self) -> None:
        hp = HomeostaticPlasticity(target_rate_hz=10.0)
        new = hp.update_threshold(256, observed_rate_hz=10.0, dt_ms=100.0)
        assert new == 256

    def test_too_fast_increases_threshold(self) -> None:
        hp = HomeostaticPlasticity(target_rate_hz=10.0, tau_homeo_ms=1000.0)
        new = hp.update_threshold(256, observed_rate_hz=50.0, dt_ms=1000.0)
        assert new > 256

    def test_too_slow_decreases_threshold(self) -> None:
        hp = HomeostaticPlasticity(target_rate_hz=10.0, tau_homeo_ms=1000.0)
        new = hp.update_threshold(256, observed_rate_hz=1.0, dt_ms=1000.0)
        assert new < 256

    def test_bounded(self) -> None:
        hp = HomeostaticPlasticity(max_threshold_q88=512, min_threshold_q88=64)
        new = hp.update_threshold(500, observed_rate_hz=1000.0, dt_ms=10000.0)
        assert new <= 512
        new = hp.update_threshold(70, observed_rate_hz=0.0, dt_ms=10000.0)
        assert new >= 64


# ── BioHybridFrameResult — dataclass + mapping dual interface ──────────


class TestBioHybridFrameResult:
    """The packet returned by ``BioHybridSession.process_frame`` must be
    both a typed dataclass (new callers) and a read-only mapping view
    (legacy callers that did ``result["round"]``). Both surfaces carry
    identical data; the mapping wraps the dataclass, not a shadow dict.
    """

    def _make(self, **overrides: Any) -> BioHybridFrameResult:
        base: dict[str, Any] = dict(
            round=3,
            num_spikes=12,
            num_aer_events=12,
            num_bitstreams=4,
            num_opto_pulses=2,
            latency_us=1234.5,
            health={"score": 0.95},
            spikes=[],
            aer_events=[],
            bitstreams={},
            opto_pulses=[],
        )
        base.update(overrides)
        return BioHybridFrameResult(**base)

    def test_attribute_access(self) -> None:
        r = self._make()
        assert r.round == 3
        assert r.latency_us == pytest.approx(1234.5)
        assert r.health["score"] == pytest.approx(0.95)

    def test_dict_subscript_matches_attribute(self) -> None:
        r = self._make()
        assert r["round"] == r.round
        assert r["latency_us"] == r.latency_us
        assert r["health"] is r.health  # same object, not a copy

    def test_contains_reports_field_names(self) -> None:
        r = self._make()
        assert "round" in r
        assert "latency_us" in r
        assert "not_a_field" not in r
        assert 42 not in r  # non-string keys are not fields

    def test_unknown_key_raises_keyerror(self) -> None:
        r = self._make()
        with pytest.raises(KeyError, match="nope"):
            _ = r["nope"]

    def test_private_attribute_hidden_from_mapping(self) -> None:
        # Mapping view must not leak Python dunder / private names.
        r = self._make()
        with pytest.raises(KeyError):
            _ = r["__class__"]

    def test_keys_returns_declared_fields(self) -> None:
        r = self._make()
        assert set(r.keys()) == {
            "round",
            "num_spikes",
            "num_aer_events",
            "num_bitstreams",
            "num_opto_pulses",
            "latency_us",
            "health",
            "spikes",
            "aer_events",
            "bitstreams",
            "opto_pulses",
        }


# ── mea_fitness_hook — evo_substrate fitness adaptor ───────────────────


class TestMEAFitnessHook:
    """``mea_fitness_hook`` converts MEA spike dynamics into the
    ``{"accuracy", "energy_mw", "latency_ms"}`` triple consumed by the
    evo_substrate ``ReplicationEngine`` fitness function.
    """

    def test_empty_spikes_returns_floor(self) -> None:
        r = mea_fitness_hook([])
        assert r == {"accuracy": 0.1, "energy_mw": 0.0, "latency_ms": 0.0}

    def test_near_target_rate_scores_high(self) -> None:
        # 10 spikes on a single channel, target_rate=10 → mean_rate = 10 → accuracy 0.99 ceiling.
        spikes = [
            DetectedSpike(channel=0, timestamp_s=i * 0.01, amplitude_uv=-40.0) for i in range(10)
        ]
        r = mea_fitness_hook(spikes, target_rate=10.0)
        assert r["accuracy"] == pytest.approx(0.99, abs=1e-9)

    def test_off_target_rate_penalised(self) -> None:
        # 100 spikes on one channel, target 10 → rate_error ratio = 9 → accuracy floor.
        spikes = [
            DetectedSpike(channel=0, timestamp_s=i * 0.001, amplitude_uv=-40.0) for i in range(100)
        ]
        r = mea_fitness_hook(spikes, target_rate=10.0)
        assert r["accuracy"] == pytest.approx(0.1, abs=1e-9)

    def test_energy_scales_with_spike_count(self) -> None:
        spikes = [DetectedSpike(channel=0, timestamp_s=0.0, amplitude_uv=-40.0)] * 20
        r = mea_fitness_hook(spikes)
        assert r["energy_mw"] == pytest.approx(20 * 0.5)

    def test_duration_converts_counts_to_rates(self) -> None:
        spikes = [
            DetectedSpike(channel=0, timestamp_s=i * 0.05, amplitude_uv=-40.0) for i in range(20)
        ]
        r = mea_fitness_hook(spikes, target_rate=10.0, duration_s=2.0)
        assert r["accuracy"] == pytest.approx(0.99, abs=1e-9)

    def test_target_rate_zero_returns_floor(self) -> None:
        spikes = [DetectedSpike(channel=0, timestamp_s=0.0, amplitude_uv=-40.0)]
        r = mea_fitness_hook(spikes, target_rate=0.0)
        assert r["accuracy"] == pytest.approx(0.1, abs=1e-9)

    def test_channel_key_used_not_channel_id(self) -> None:
        # Regression guard: previous implementation accessed ``s.channel_id``
        # which doesn't exist on DetectedSpike and raised AttributeError on
        # any non-empty input.
        spikes = [
            DetectedSpike(channel=0, timestamp_s=0.0, amplitude_uv=-40.0),
            DetectedSpike(channel=1, timestamp_s=0.0, amplitude_uv=-40.0),
        ]
        r = mea_fitness_hook(spikes)
        assert {"accuracy", "energy_mw", "latency_ms"} == set(r.keys())

    def test_latency_uses_measured_closed_loop_value(self) -> None:
        spikes = [DetectedSpike(channel=0, timestamp_s=0.25, amplitude_uv=-40.0)]
        r = mea_fitness_hook(spikes, measured_latency_ms=3.75)
        assert r["latency_ms"] == pytest.approx(3.75)

    def test_latency_uses_first_response_after_stimulus(self) -> None:
        spikes = [
            DetectedSpike(channel=0, timestamp_s=0.090, amplitude_uv=-40.0),
            DetectedSpike(channel=0, timestamp_s=0.125, amplitude_uv=-40.0),
            DetectedSpike(channel=1, timestamp_s=0.140, amplitude_uv=-40.0),
        ]
        r = mea_fitness_hook(spikes, stimulus_time_s=0.100)
        assert r["latency_ms"] == pytest.approx(25.0)

    def test_latency_zero_when_no_spike_follows_stimulus(self) -> None:
        # Every spike precedes the stimulus, so there is no causal response.
        spikes = [DetectedSpike(channel=0, timestamp_s=0.05, amplitude_uv=-40.0)]
        r = mea_fitness_hook(spikes, stimulus_time_s=0.1)
        assert r["latency_ms"] == 0.0

    def test_latency_non_finite_timestamp_raises(self) -> None:
        spikes = [DetectedSpike(channel=0, timestamp_s=float("inf"), amplitude_uv=-40.0)]
        with pytest.raises(ValueError, match="timestamps must be finite"):
            mea_fitness_hook(spikes)

    def test_response_latency_empty_spikes_without_measured_is_zero(self) -> None:
        from sc_neurocore.bioware.bioware import _mea_response_latency_ms

        assert _mea_response_latency_ms([], stimulus_time_s=None, measured_latency_ms=None) == 0.0

    def test_latency_without_stimulus_uses_first_spike_timestamp(self) -> None:
        spikes = [
            DetectedSpike(channel=0, timestamp_s=0.006, amplitude_uv=-40.0),
            DetectedSpike(channel=1, timestamp_s=0.014, amplitude_uv=-40.0),
        ]
        r = mea_fitness_hook(spikes)
        assert r["latency_ms"] == pytest.approx(6.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"measured_latency_ms": -1.0},
            {"stimulus_time_s": float("nan")},
            {"duration_s": 0.0},
        ],
    )
    def test_rejects_invalid_fitness_timing_parameters(self, kwargs: dict[str, float]) -> None:
        spikes = [DetectedSpike(channel=0, timestamp_s=0.0, amplitude_uv=-40.0)]
        with pytest.raises(ValueError):
            mea_fitness_hook(spikes, **cast(Any, kwargs))
