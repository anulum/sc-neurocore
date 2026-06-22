# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Sensor Fusion Tests

import numpy as np

from sc_neurocore.fusion.sensor_fusion import (
    BitstreamDecorrelator,
    CochleaAdapter,
    CrossModalAttention,
    DVSAdapter,
    EventStream,
    FusionEnergyEstimator,
    FusionVerilogEmitter,
    HDCBinding,
    IMUAdapter,
    SensorFusionLayer,
    SensorModality,
    TactileAdapter,
    TemporalAligner,
)


def _make_stream(
    modality: SensorModality,
    n_events: int = 100,
    seed: int = 0,
) -> EventStream:
    rng = np.random.default_rng(seed)
    return EventStream(
        modality=modality,
        timestamps=np.sort(rng.integers(0, 1_000_000, n_events)).astype(np.float64),
        addresses=rng.integers(0, 64, n_events),
        polarities=rng.choice([-1, 1], n_events),
    )


# ── EventStream Tests ────────────────────────────────────────────────


class TestEventStream:
    def test_num_events(self):
        s = _make_stream(SensorModality.DVS, n_events=50)
        assert s.num_events == 50

    def test_duration(self):
        s = _make_stream(SensorModality.DVS, n_events=100)
        assert s.duration_us > 0

    def test_event_rate(self):
        s = _make_stream(SensorModality.DVS, n_events=100)
        assert s.event_rate > 0

    def test_to_bitstream_shape(self):
        s = _make_stream(SensorModality.DVS, n_events=50)
        bs = s.to_bitstream(length=256, num_channels=32)
        assert bs.shape == (32, 256)

    def test_empty_stream(self):
        s = EventStream(
            modality=SensorModality.DVS,
            timestamps=np.array([]),
            addresses=np.array([]),
            polarities=np.array([]),
        )
        assert s.num_events == 0
        assert s.duration_us == 0.0
        bs = s.to_bitstream(128, 16)
        assert np.sum(bs) == 0


# ── BitstreamDecorrelator Tests ──────────────────────────────────────


class TestBitstreamDecorrelator:
    def test_decorrelate_produces_different_streams(self):
        dec = BitstreamDecorrelator(seed=42)
        a = np.ones((8, 64), dtype=np.uint8)
        b = np.ones((8, 64), dtype=np.uint8)
        result = dec.decorrelate([a, b])
        assert not np.array_equal(result[0], result[1])

    def test_decorrelate_preserves_shape(self):
        dec = BitstreamDecorrelator(seed=42)
        a = np.ones((16, 128), dtype=np.uint8)
        result = dec.decorrelate([a])
        assert result[0].shape == (16, 128)

    def test_sobol_method(self):
        dec = BitstreamDecorrelator(seed=42)
        a = np.ones((4, 32), dtype=np.uint8)
        result = dec.decorrelate([a], method="sobol")
        assert result[0].shape == (4, 32)

    def test_scc_returns_bounded_value(self):
        dec = BitstreamDecorrelator(seed=42)
        rng = np.random.default_rng(0)
        a = rng.integers(0, 2, 100, dtype=np.uint8)
        b = rng.integers(0, 2, 100, dtype=np.uint8)
        scc = dec.measure_scc(a, b)
        assert -1.0 <= scc <= 1.0

    def test_scc_identical_streams(self):
        dec = BitstreamDecorrelator()
        a = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        scc = dec.measure_scc(a, a)
        assert abs(scc - 1.0) < 0.01

    def test_seed_zero_collides_and_resets_to_one(self):
        # base_seed 0 makes the i=0 mask seed land on 0, which the generator
        # must bump to 1 (a zero LFSR seed produces a degenerate all-zero mask).
        dec = BitstreamDecorrelator(seed=0)
        stream = np.ones((2, 4), dtype=np.uint8)
        result = dec.decorrelate([stream])
        assert result[0].shape == (2, 4)

    def test_scc_independent_streams_hit_numerator_floor(self):
        # Two all-zero streams give pa=pb=p_and=0, so the numerator collapses
        # to the |num|<eps floor and the coefficient is defined as 0.
        dec = BitstreamDecorrelator()
        zeros = np.zeros(8, dtype=np.float64)
        assert dec.measure_scc(zeros, zeros) == 0.0

    def test_scc_degenerate_denominator_returns_zero(self):
        # A non-binary stream breaks the bitstream invariant p_and<=min(pa,pb):
        # for a=[1.5,0.5] (pa=1.0) the denominator min(pa,pb)-pa*pb is exactly 0
        # while the numerator stays positive, exercising the |denom|<eps floor.
        dec = BitstreamDecorrelator()
        degenerate = np.array([1.5, 0.5], dtype=np.float64)
        assert dec.measure_scc(degenerate, degenerate) == 0.0


# ── CrossModalAttention Tests ────────────────────────────────────────


class TestCrossModalAttention:
    def test_attend_preserves_shape(self):
        attn = CrossModalAttention(num_channels=8, seed=42)
        q = np.ones((8, 32), dtype=np.uint8)
        k = np.ones((8, 32), dtype=np.uint8)
        v = np.ones((8, 32), dtype=np.uint8)
        result = attn.attend(q, k, v)
        assert result.shape == (8, 32)

    def test_attend_zero_query_zero_output(self):
        attn = CrossModalAttention(num_channels=4, seed=42)
        q = np.zeros((4, 16), dtype=np.uint8)
        k = np.ones((4, 16), dtype=np.uint8)
        v = np.ones((4, 16), dtype=np.uint8)
        result = attn.attend(q, k, v)
        assert np.sum(result) == 0

    def test_sc_and_multiplication(self):
        attn = CrossModalAttention(num_channels=4)
        a = np.array([[1, 0, 1, 1]], dtype=np.uint8)
        b = np.array([[1, 1, 0, 1]], dtype=np.uint8)
        result = attn._sc_and(a, b)
        np.testing.assert_array_equal(result, [[1, 0, 0, 1]])


# ── SensorFusionLayer Tests ──────────────────────────────────────────


class TestSensorFusionLayer:
    def test_fuse_two_streams(self):
        layer = SensorFusionLayer(num_channels=16, bitstream_length=128, seed=42)
        s1 = _make_stream(SensorModality.DVS, 50, seed=0)
        s2 = _make_stream(SensorModality.TACTILE, 50, seed=1)
        fused, metrics = layer.fuse([s1, s2])
        assert fused.shape == (16, 128)
        assert metrics.num_streams == 2
        assert metrics.total_events == 100

    def test_fuse_empty_list(self):
        layer = SensorFusionLayer()
        fused, metrics = layer.fuse([])
        assert np.sum(fused) == 0
        assert metrics.num_streams == 0

    def test_fuse_single_stream(self):
        layer = SensorFusionLayer(num_channels=8, bitstream_length=64, seed=42)
        s = _make_stream(SensorModality.COCHLEA, 30, seed=0)
        fused, metrics = layer.fuse([s])
        assert fused.shape == (8, 64)
        assert metrics.num_streams == 1

    def test_modality_weighting(self):
        layer = SensorFusionLayer(num_channels=8, bitstream_length=64, seed=42)
        layer.set_weight(SensorModality.DVS, 0.1)
        s = _make_stream(SensorModality.DVS, 100, seed=0)
        fused_weighted, _ = layer.fuse([s], use_attention=False)

        layer2 = SensorFusionLayer(num_channels=8, bitstream_length=64, seed=42)
        fused_full, _ = layer2.fuse([s], use_attention=False)

        assert np.sum(fused_weighted) <= np.sum(fused_full)

    def test_latency_recorded(self):
        layer = SensorFusionLayer(num_channels=8, bitstream_length=64, seed=42)
        s = _make_stream(SensorModality.DVS, 50)
        _, metrics = layer.fuse([s])
        assert metrics.latency_us > 0.0

    def test_cross_modal_scc_bounded(self):
        layer = SensorFusionLayer(num_channels=8, bitstream_length=128, seed=42)
        s1 = _make_stream(SensorModality.DVS, 80, seed=0)
        s2 = _make_stream(SensorModality.TACTILE, 80, seed=1)
        _, metrics = layer.fuse([s1, s2])
        assert -1.0 <= metrics.cross_modal_scc <= 1.0

    def test_three_modality_fusion(self):
        layer = SensorFusionLayer(num_channels=8, bitstream_length=64, seed=42)
        streams = [
            _make_stream(SensorModality.DVS, 30, seed=0),
            _make_stream(SensorModality.TACTILE, 30, seed=1),
            _make_stream(SensorModality.COCHLEA, 30, seed=2),
        ]
        fused, metrics = layer.fuse(streams)
        assert metrics.num_streams == 3
        assert fused.shape == (8, 64)

    def test_fuse_without_attention(self):
        layer = SensorFusionLayer(num_channels=8, bitstream_length=64, seed=42)
        s1 = _make_stream(SensorModality.DVS, 50, seed=0)
        s2 = _make_stream(SensorModality.TACTILE, 50, seed=1)
        fused, metrics = layer.fuse([s1, s2], use_attention=False)
        assert fused.shape == (8, 64)
        assert metrics.fused_popcount >= 0


# ── HDCBinding Tests ─────────────────────────────────────────────────


class TestHDCBinding:
    def test_hypervector_dimension(self):
        hdc = HDCBinding(dim=2048)
        hv = hdc.get_hypervector("dvs")
        assert len(hv) == 2048

    def test_hypervector_deterministic(self):
        hdc = HDCBinding(dim=512, seed=42)
        a = hdc.get_hypervector("test")
        b = hdc.get_hypervector("test")
        np.testing.assert_array_equal(a, b)

    def test_bind_is_self_inverse(self):
        hdc = HDCBinding(dim=1024, seed=42)
        a = hdc.get_hypervector("a")
        b = hdc.get_hypervector("b")
        bound = hdc.bind(a, b)
        unbound = hdc.bind(bound, b)
        np.testing.assert_array_equal(unbound, a)

    def test_bundle_majority_vote(self):
        hdc = HDCBinding(dim=1024, seed=42)
        a = np.ones(1024, dtype=np.uint8)
        b = np.ones(1024, dtype=np.uint8)
        c = np.zeros(1024, dtype=np.uint8)
        result = hdc.bundle([a, b, c])
        assert np.sum(result) == 1024  # majority is 1

    def test_bundle_empty_returns_zero_hypervector(self):
        hdc = HDCBinding(dim=64)
        result = hdc.bundle([])
        assert result.shape == (64,)
        assert np.sum(result) == 0

    def test_similarity_identical(self):
        hdc = HDCBinding(dim=512, seed=42)
        a = hdc.get_hypervector("x")
        assert hdc.similarity(a, a) == 1.0

    def test_similarity_random_near_half(self):
        hdc = HDCBinding(dim=4096, seed=42)
        a = hdc.get_hypervector("x")
        b = hdc.get_hypervector("y")
        sim = hdc.similarity(a, b)
        assert 0.3 < sim < 0.7

    def test_encode_stream(self):
        hdc = HDCBinding(dim=1024, seed=42)
        s = _make_stream(SensorModality.DVS, 50, seed=0)
        hv = hdc.encode_stream(s)
        assert len(hv) == 1024
        assert hv.dtype == np.uint8

    def test_different_modalities_different_encoding(self):
        hdc = HDCBinding(dim=1024, seed=42)
        s1 = _make_stream(SensorModality.DVS, 50, seed=0)
        s2 = _make_stream(SensorModality.COCHLEA, 50, seed=0)
        hv1 = hdc.encode_stream(s1)
        hv2 = hdc.encode_stream(s2)
        assert not np.array_equal(hv1, hv2)


# ── DVSAdapter Tests ─────────────────────────────────────────────────


class TestDVSAdapter:
    def test_encode_events(self):
        ts = np.arange(10, dtype=np.float64) * 1000
        x = np.arange(10) % 128
        y = np.arange(10) % 128
        pol = np.ones(10, dtype=np.int8)
        stream = DVSAdapter.encode_events(ts, x, y, pol)
        assert stream.modality == SensorModality.DVS
        assert stream.num_events == 10
        assert "resolution" in stream.metadata

    def test_address_encoding(self):
        ts = np.array([0.0, 1000.0])
        x = np.array([5, 10])
        y = np.array([3, 7])
        pol = np.array([1, -1], dtype=np.int8)
        stream = DVSAdapter.encode_events(ts, x, y, pol, resolution=(128, 128))
        assert stream.addresses[0] == (3 * 128 + 5) % (128 * 128)


# ── CochleaAdapter Tests ────────────────────────────────────────────


class TestCochleaAdapter:
    def test_freq_to_channel_boundaries(self):
        coch = CochleaAdapter(num_channels=64)
        assert coch.freq_to_channel(10.0) == 0
        assert coch.freq_to_channel(25000.0) == 63

    def test_freq_to_channel_mid(self):
        coch = CochleaAdapter(num_channels=64)
        ch = coch.freq_to_channel(1000.0)
        assert 0 < ch < 63

    def test_log_scale_ordering(self):
        coch = CochleaAdapter(num_channels=64)
        ch_low = coch.freq_to_channel(100.0)
        ch_mid = coch.freq_to_channel(1000.0)
        ch_high = coch.freq_to_channel(10000.0)
        assert ch_low < ch_mid < ch_high

    def test_encode_spikes(self):
        ts = np.arange(5, dtype=np.float64) * 100
        freqs = np.array([100.0, 500.0, 1000.0, 5000.0, 10000.0])
        coch = CochleaAdapter(num_channels=32)
        stream = coch.encode_spikes(ts, freqs)
        assert stream.modality == SensorModality.COCHLEA
        assert stream.num_events == 5


# ── TactileAdapter Tests ────────────────────────────────────────────


class TestTactileAdapter:
    def test_encode_pressure(self):
        ts = np.arange(4, dtype=np.float64) * 100
        taxels = np.array([0, 1, 2, 3])
        pressures = np.array([0.5, 0.05, 0.8, 0.01])
        stream = TactileAdapter.encode_pressure(ts, taxels, pressures, threshold=0.1)
        assert stream.modality == SensorModality.TACTILE
        assert stream.num_events == 4
        assert stream.polarities[0] == 1  # above threshold
        assert stream.polarities[1] == -1  # below threshold


# ── IMUAdapter Tests ─────────────────────────────────────────────────


class TestIMUAdapter:
    def test_encode_angular_rate(self):
        ts = np.arange(10, dtype=np.float64) * 100
        axes = np.zeros(10, dtype=np.int64)
        rates = np.array([10, 3, -20, 2, 15, -1, 30, 4, -8, 0], dtype=np.float64)
        stream = IMUAdapter.encode_angular_rate(ts, axes, rates, deadzone_dps=5.0)
        assert stream.modality == SensorModality.PROPRIOCEPTIVE
        assert stream.num_events < 10  # some filtered by deadzone

    def test_deadzone_filters_small(self):
        ts = np.arange(5, dtype=np.float64) * 100
        axes = np.zeros(5, dtype=np.int64)
        rates = np.array([1, 2, 3, 4, 100], dtype=np.float64)
        stream = IMUAdapter.encode_angular_rate(ts, axes, rates, deadzone_dps=50.0)
        assert stream.num_events == 1  # only 100 > 50


# ── TemporalAligner Tests ───────────────────────────────────────────


class TestTemporalAligner:
    def test_align_overlapping(self):
        aligner = TemporalAligner(window_us=1000.0)
        s1 = EventStream(
            SensorModality.DVS,
            timestamps=np.array([100, 200, 300, 400, 500], dtype=np.float64),
            addresses=np.arange(5),
            polarities=np.ones(5, dtype=np.int8),
        )
        s2 = EventStream(
            SensorModality.TACTILE,
            timestamps=np.array([200, 300, 400, 500, 600], dtype=np.float64),
            addresses=np.arange(5),
            polarities=np.ones(5, dtype=np.int8),
        )
        aligned = aligner.align([s1, s2])
        assert len(aligned) == 2
        for a in aligned:
            assert float(a.timestamps[0]) >= 200
            assert float(a.timestamps[-1]) <= 500

    def test_slice_windows(self):
        aligner = TemporalAligner(window_us=200.0)
        s = EventStream(
            SensorModality.DVS,
            timestamps=np.array([0, 100, 200, 300, 400, 500, 600], dtype=np.float64),
            addresses=np.arange(7),
            polarities=np.ones(7, dtype=np.int8),
        )
        windows = aligner.slice_windows(s)
        assert len(windows) >= 3

    def test_empty_alignment(self):
        aligner = TemporalAligner()
        assert aligner.align([]) == []

    def test_align_non_overlapping_returns_streams_unchanged(self):
        # Streams whose active spans do not overlap give t_min >= t_max, so
        # there is no common window and the originals are returned as-is.
        aligner = TemporalAligner(window_us=1000.0)
        early = EventStream(
            SensorModality.DVS,
            timestamps=np.array([100, 200], dtype=np.float64),
            addresses=np.arange(2),
            polarities=np.ones(2, dtype=np.int8),
        )
        late = EventStream(
            SensorModality.TACTILE,
            timestamps=np.array([300, 400], dtype=np.float64),
            addresses=np.arange(2),
            polarities=np.ones(2, dtype=np.int8),
        )
        aligned = aligner.align([early, late])
        assert aligned == [early, late]

    def test_slice_windows_single_event_returns_whole_stream(self):
        # A stream with fewer than two events cannot be windowed and is passed
        # through as a single window.
        aligner = TemporalAligner(window_us=200.0)
        s = EventStream(
            SensorModality.DVS,
            timestamps=np.array([100.0], dtype=np.float64),
            addresses=np.arange(1),
            polarities=np.ones(1, dtype=np.int8),
        )
        windows = aligner.slice_windows(s)
        assert windows == [s]


# ── FusionVerilogEmitter Tests ───────────────────────────────────────


class TestFusionVerilogEmitter:
    def test_emit_contains_module(self):
        sv = FusionVerilogEmitter.emit()
        assert "module sc_multimodal_fusion" in sv
        assert "endmodule" in sv

    def test_emit_custom_streams(self):
        sv = FusionVerilogEmitter.emit(num_streams=6, bitstream_width=32)
        assert "STREAMS      = 6" in sv
        assert "BITSTREAM_W  = 32" in sv

    def test_emit_attention_mode(self):
        sv = FusionVerilogEmitter.emit(use_attention=True)
        assert "SC-AND" in sv or "coincidence" in sv

    def test_emit_or_mode(self):
        sv = FusionVerilogEmitter.emit(use_attention=False)
        assert "OR fusion" in sv

    def test_emit_custom_name(self):
        sv = FusionVerilogEmitter.emit(module_name="my_fusion")
        assert "module my_fusion" in sv

    def test_lfsr_decorrelation_present(self):
        sv = FusionVerilogEmitter.emit()
        assert "lfsr" in sv.lower()
        assert "decorr" in sv


# ── FusionEnergyEstimator Tests ──────────────────────────────────────


class TestFusionEnergyEstimator:
    def test_basic_estimate(self):
        est = FusionEnergyEstimator(tech_node_nm=28)
        result = est.estimate(num_streams=4, num_channels=64, bitstream_length=256)
        assert result.total_uw > 0
        assert result.decorrelation_uw > 0
        assert result.attention_uw > 0
        assert result.routing_uw > 0

    def test_no_attention_lower_energy(self):
        est = FusionEnergyEstimator(tech_node_nm=28)
        with_attn = est.estimate(4, 64, 256, use_attention=True)
        without_attn = est.estimate(4, 64, 256, use_attention=False)
        assert without_attn.total_uw < with_attn.total_uw

    def test_sub_mw_for_small_config(self):
        est = FusionEnergyEstimator(tech_node_nm=7)
        result = est.estimate(
            num_streams=2, num_channels=4, bitstream_length=16, use_attention=False
        )
        assert result.total_mw < 1.0

    def test_scales_with_tech_node(self):
        est_7nm = FusionEnergyEstimator(tech_node_nm=7)
        est_28nm = FusionEnergyEstimator(tech_node_nm=28)
        r7 = est_7nm.estimate(4, 64, 256)
        r28 = est_28nm.estimate(4, 64, 256)
        assert r7.total_uw < r28.total_uw

    def test_total_mw_conversion(self):
        est = FusionEnergyEstimator()
        result = est.estimate(2, 8, 64)
        assert abs(result.total_mw - result.total_uw / 1000.0) < 1e-10
